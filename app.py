import cv2
import pygame
import sys

import numpy as np

from keras.models import load_model
from pygame.locals import *

MODEL = load_model("my_model.h5")

BOUNDARY = 5
WINDOW_SIZE_X = 800
WINDOW_SIZE_Y = 600

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

IMAGE_SAVE = False
PREDICT = True

pygame.init()

DISPLAY_SURFACE = pygame.display.set_mode((WINDOW_SIZE_X, WINDOW_SIZE_Y))
white_int = DISPLAY_SURFACE.map_rgb(WHITE)
pygame.display.set_caption('Digit Recognizer')

is_writing = False
num_x_coords = []
num_y_coords = []
img_count = 1

while True:
    
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        
        elif event.type == MOUSEMOTION and is_writing:
            x_coord, y_coord = event.pos
            pygame.draw.circle(DISPLAY_SURFACE, WHITE, (x_coord, y_coord), 4, 0)
            num_x_coords.append(x_coord)
            num_y_coords.append(y_coord)

        elif event.type == MOUSEBUTTONDOWN:
            is_writing = True

        elif event.type == MOUSEBUTTONUP:
            is_writing = False
            num_x_coords = sorted(num_x_coords)
            num_y_coords = sorted(num_y_coords)

            rect_min_x, rect_max_x = max(num_x_coords[0] - BOUNDARY, 0), min(WINDOW_SIZE_X - num_x_coords[-1] + BOUNDARY)
            rect_min_y, rect_max_y = max(num_y_coords[0] - BOUNDARY, 0), min(WINDOW_SIZE_Y - num_y_coords[-1] + BOUNDARY)

            num_x_coords = []
            num_y_coords = []
            img_arr = np.array(pygame.PixelArray(DISPLAY_SURFACE))
            
            if IMAGE_SAVE:
                cv2.imwrite("image.png")
                img_count += 1
            
            if PREDICT:
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, (10, 10), "constant", constant_values=0)
                image = cv2.resize(image, (28, 28)) / white_int
                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))]).title()
                pygame.draw.rect(DISPLAY_SURFACE, RED, (rect_min_x, rect_min_y, rect_max_x - rect_min_x, rect_max_y - rect_min_y), 3)

        elif event.type == KEYDOWN:
            if event.unicode == "N":
                DISPLAY_SURFACE.fill(BLACK)
    
    pygame.display.update()
