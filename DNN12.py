import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import requests
from PIL import Image
from io import BytesIO

categories=['dandelion', 'daisy', 'sunflower', 'tulip', 'rose']
dire='../input/flowers/flowers/'

features=[]
for i in categories:
    path=os.path.join(dire,i)
    num_classes=categories.index(i)
    for img in os.listdir(path):
        if img.endswith('.jpg'):
            
            img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_COLOR)
            img_array=cv2.resize(img_array,(150,150))
            features.append([img_array,num_classes])
X=[]
y=[]
for i,j in features:
    X.append(i)
    y.append(j)





X=np.array(X).reshape(-1,150,150,3)/255.0

list_dandelion=len([i for i in y if i==0])
list_daisy=len([i for i in y if i==1])
list_sunflower=len([i for i in y if i==2])
list_tulip=len([ i for i in y if i==3])
list_rose=len([i for i in y if i==4])

list_species=[list_dandelion,list_daisy,list_sunflower,list_tulip,list_rose]

y=to_categorical(y)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=12)


# =============================================================================
# model = Sequential()
# 
# model.add(Conv2D(32, (3, 3), input_shape=(150,150,3)))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(2, 2, padding="same"))
# model.add(Dropout(0.2))
# 
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(2, 2, padding="same"))
# model.add(Dropout(0.2))
# 
# model.add(Conv2D(128, (3, 3)))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(2, 2, padding="same"))
# model.add(Dropout(0.2))
# 
# model.add(Flatten())
# model.add(Dense(512, activation="relu"))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation="relu"))
# model.add(Dropout(0.2))
# model.add(Dense(5, activation="softmax"))
# 
# epochs=100
# batch_size=128
# 
# 
# red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)
# 
# 
# datagen = ImageDataGenerator(
#         featurewise_center=False,  # set input mean to 0 over the dataset
#         samplewise_center=False,  # set each sample mean to 0
#         featurewise_std_normalization=False,  # divide inputs by std of the dataset
#         samplewise_std_normalization=False,  # divide each input by its std
#         zca_whitening=False,  # apply ZCA whitening
#         rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
#         zoom_range = 0.1, # Randomly zoom image 
#         width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
#         height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
#         horizontal_flip=True,  # randomly flip images
#         vertical_flip=False)  # randomly flip images
# 
# 
# datagen.fit(x_train)
# 
# model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
# 
# History = model.fit_generator(datagen.flow(x_train,y_train, batch_size=128),
#                               epochs = epochs, validation_data = (x_test,y_test),
#                               verbose = 1, steps_per_epoch=x_train.shape[0] // 128)
# 
# model.save("model12.h5")
# =============================================================================

model = keras.models.load_model("model12.h5")

# =============================================================================
# custom_path = '../custom_input/'
# 
# c_img_array=cv2.imread(custom_path,cv2.IMREAD_COLOR)
#     
# # =============================================================================
# #     c_img_array=cv2.resize(c_img_array,(150,150))
# # =============================================================================
# fig,ax=plt.subplots(5,2)
# fig.set_size_inches(15,15)
# for i in range(5):
#     for j in range (2):
#         l=np.random.randint(0,len(c_img_array))
#         ax[i,j].imshow(X[l])
# plt.axis('off')        
# plt.tight_layout()
# 
# c_img_array=np.array(c_img_array).reshape(-1,150,150,3)/255.0
# 
# preds2=model.predict(c_img_array)
# predictions2=np.argmax(preds2,axis=1)
# 
# count=0
# fig,ax=plt.subplots(4,2)
# fig.set_size_inches(15,15)
# for i in range (4):
#     for j in range (2):
#         ax[i,j].imshow(c_img_array[count])
#         ax[i,j].set_title("Predicted Flower : "+ categories[predictions2[count]] +"\n")
#         plt.tight_layout()
#         count+=1
# =============================================================================
def process_image(url):
    response=requests.get(url)
    img=Image.open(BytesIO(response.content))
    fix,ax=plt.subplots(1,3,figsize=(15,20))
    ax[0].imshow(img)
    ax[0].set_title('image')
    
    #grayscale and normalization
    img=np.array(img)
    img=cv2.cvtColor(img,cv2.IMREAD_COLOR)
    print(img.shape)
    img=img/255.0
    ax[1].imshow(img)
    ax[1].set_title('color image')
    
    #resizing
    img=cv2.resize(img,(150,150))
    print(img.shape)
    ax[2].imshow(img)
    ax[2].set_title('predicted image')
    plt.tight_layout()
    img=np.expand_dims(img,axis=0)
    #making it model ready
    
    print(img.shape)
    return img

def predict(url):
    img=process_image(url)
    label=model.predict(img)
    final_1=np.argmax(label,axis=1)[0]
    plt.xlabel(categories[final_1])
    return categories[final_1]


predict('https://i.pinimg.com/originals/58/32/e7/5832e7928deb5fa40e09cd552db010f3.jpg')