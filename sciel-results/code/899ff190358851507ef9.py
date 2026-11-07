import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with a 10-term mixed basis:
    - 5 Chebyshev polynomial terms (well-conditioned on [-1, 1])
    - 5 trigonometric terms (captures periodic/oscillatory structure)
    
    Basis:
    0: 1
    1: T1(x) = x
    2: T1(y) = y
    3: T2(x) = 2x^2 - 1
    4: T2(y) = 2y^2 - 1
    5: sin(x)
    6: sin(y)
    7: cos(x)
    8: cos(y)
    9: sin(x) * sin(y)
    """
    # Chebyshev Polynomials of the First Kind
    T1_x = x
    T1_y = y
    T2_x = 2.0 * x * x - 1.0
    T2_y = 2.0 * y * y - 1.0
    
    # Trigonometric terms
    sin_x = np.sin(x)
    sin_y = np.sin(y)
    cos_x = np.cos(x)
    cos_y = np.cos(y)
    sin_x_sin_y = sin_x * sin_y
    
    # Linear combination of basis functions
    f = (params[0] * 1.0 +
         params[1] * T1_x +
         params[2] * T1_y +
         params[3] * T2_x +
         params[4] * T2_y +
         params[5] * sin_x +
         params[6] * sin_y +
         params[7] * cos_x +
         params[8] * cos_y +
         params[9] * sin_x_sin_y)
         
    return f
