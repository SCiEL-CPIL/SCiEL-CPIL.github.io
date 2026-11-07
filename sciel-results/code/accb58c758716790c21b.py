import numpy as np
import time

_START_TIME = time.time()

def equation(x, y, params):
    # Deadline guard: safety margin for the 150s limit
    if time.time() - _START_TIME > 145:
        return np.zeros_like(x)
    
    # Mixed polynomial + trigonometric basis for flexible function approximation
    # Combines low-order polynomials with multiple frequency sine/cosine terms
    # to capture both smooth trends and oscillatory components
    
    # Polynomial features
    x2 = x * x
    y2 = y * y
    xy = x * y
    
    # Trigonometric features at multiple frequencies
    sx1 = np.sin(np.pi * x)
    cx1 = np.cos(np.pi * x)
    sy1 = np.sin(np.pi * y)
    cy1 = np.cos(np.pi * y)
    
    sx2 = np.sin(2 * np.pi * x)
    cx2 = np.cos(2 * np.pi * x)
    sy2 = np.sin(2 * np.pi * y)
    cy2 = np.cos(2 * np.pi * y)
    
    # 10 basis functions: constant, x, y, x^2, y^2, xy, sin(pi*x), cos(pi*y), sin(2pi*x), sin(2pi*y)
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x2
    b4 = y2
    b5 = xy
    b6 = sx1
    b7 = cy1
    b8 = sx2
    b9 = sy2
    
    result = (params[0]*b0 + params[1]*b1 + params[2]*b2 + params[3]*b3 +
              params[4]*b4 + params[5]*b5 + params[6]*b6 + params[7]*b7 +
              params[8]*b8 + params[9]*b9)
    
    return result
