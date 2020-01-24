#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 15:36:49 2017

@author: adam
"""

import cv2
import numpy as np
from keras.models import load_model
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

video_capture = cv2.VideoCapture(0)
model = load_model('keras_model/model_5-49-0.62.hdf5')

target = ['angry','disgust','fear','happy','sad','surprised','neutral']
font = cv2.FONT_HERSHEY_SIMPLEX


counter = 0
counter1 = 0

USE_WEBCAM = True

# if (USE_WEBCAM == True):
#     cap = cv2.VideoCapture(0) # Webcam source
# else:
#     cap = cv2.VideoCapture('./test/tbbt.mp4')
# ret, frame = cap.read() 
#while True:
	   # Capture frame-by-frame
	   # ret, frame = video_capture.read(0)




    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1)

    

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2,5)
        # crop the detected face region
        face_crop = np.copy(frame[y:y+h,x:x+w])
        #face_crop = frame[y:y+h,x:x+w]
        face_crop = cv2.resize(face_crop,(48,48))
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        face_crop = face_crop.astype('float32')/255
        face_crop = np.asarray(face_crop)
        face_crop = face_crop.reshape(1, 1,face_crop.shape[0],face_crop.shape[1])
        result = target[np.argmax(model.predict(face_crop))]
        cv2.putText(frame,result,(x,y), font, 1, (200,0,0), 3, cv2.LINE_AA)
        face_crop = frame[y:y+h, x:x+w]
        #print(target)
        if (result == 'neutral'):
            cv2.imwrite("G:/Devnet/facial_emotion_recognition-master/neutral/" + "neutral"+str(counter)+".png",face_crop)
            counter = counter + 1
        elif (result == 'happy'):
            cv2.imwrite("G:/Devnet/facial_emotion_recognition-master/happy/" + "happy"+str(counter1)+".png",face_crop)
            counter1 = counter1 + 1
        elif (result == 'surprised'):
            cv2.imwrite("G:/Devnet/facial_emotion_recognition-master/surprised/" + "surprised"+str(counter1)+".png",face_crop)
            counter1 = counter1 + 1
        elif (result == 'sad'):
            cv2.imwrite("G:/Devnet/facial_emotion_recognition-master/sad/" + "sad"+str(counter1)+".png",face_crop)
            counter1 = counter1 + 1
        elif (result == 'disgust'):
            cv2.imwrite("G:/Devnet/facial_emotion_recognition-master/disgust/" + "disgust"+str(counter1)+".png",face_crop)
            counter1 = counter1 + 1
        elif (result == 'angry'):
            cv2.imwrite("G:/Devnet/facial_emotion_recognition-master/angry/" + "angry"+str(counter1)+".png",face_crop)
            counter1 = counter1 + 1

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
