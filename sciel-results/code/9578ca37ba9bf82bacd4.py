import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Tensor-product Fourier basis with fundamental frequency pi (period 2)
    # This aligns well with the [-1, 1] domain.
    # We use 10 parameters corresponding to:
    # 0: constant (1)
    # 1: cos(pi*x)
    # 2: sin(pi*x)
    # 3: cos(pi*y)
    # 4: sin(pi*y)
    # 5: cos(pi*x)*cos(pi*y)
    # 6: cos(pi*x)*sin(pi*y)
    # 7: sin(pi*x)*cos(pi*y)
    # 8: sin(pi*x)*sin(pi*y)
    # 9: cos(2*pi*x) + cos(2*pi*y)  (higher frequency symmetric term to capture curvature)
    
    # Precompute trigonometric components
    c_x = np.cos(np.pi * x)
    s_x = np.sin(np.pi * x)
    c_y = np.cos(np.pi * y)
    s_y = np.sin(np.pi * y)
    
    # Basis functions
    b0 = np.ones_like(x)
    b1 = c_x
    b2 = s_x
    b3 = c_y
    b4 = s_y
    b5 = c_x * c_y
    b6 = c_x * s_y
    b7 = s_x * c_y
    b8 = s_x * s_y
    # For the 10th term, we use a symmetric higher frequency term.
    # cos(2*pi*x) = 2*cos^2(pi*x) - 1
    # cos(2*pi*y) = 2*cos^2(pi*y) - 1
    b9 = (2 * c_x * c_x - 1) + (2 * c_y * c_y - 1)
    
    return (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)
