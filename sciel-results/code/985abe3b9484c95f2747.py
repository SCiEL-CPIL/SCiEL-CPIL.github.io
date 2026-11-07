import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a 2D Chebyshev polynomial basis up to total degree 3.
    
    Chebyshev polynomials are well-suited for approximating smooth functions on [-1, 1]
    and provide a well-conditioned basis for least-squares fitting, which helps
    scipy.optimize.minimize (BFGS) converge reliably.
    
    The basis consists of 10 terms: T_i(x) * T_j(y) where i + j <= 3.
    T_0(t) = 1
    T_1(t) = t
    T_2(t) = 2t^2 - 1
    T_3(t) = 4t^3 - 3t
    """
    # Compute Chebyshev polynomials for x and y
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2 * x**2 - 1
    T3_x = 4 * x**3 - 3 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2 * y**2 - 1
    T3_y = 4 * y**3 - 3 * y
    
    # Construct the 10 basis functions
    # Order: (i, j) such that i + j <= 3
    b0 = T0_x * T0_y  # 1
    b1 = T1_x * T0_y  # x
    b2 = T0_x * T1_y  # y
    b3 = T2_x * T0_y  # 2x^2 - 1
    b4 = T0_x * T2_y  # 2y^2 - 1
    b5 = T1_x * T1_y  # xy
    b6 = T3_x * T0_y  # 4x^3 - 3x
    b7 = T0_x * T3_y  # 4y^3 - 3y
    b8 = T2_x * T1_y  # (2x^2 - 1)y
    b9 = T1_x * T2_y  # x(2y^2 - 1)
    
    # Linear combination
    f = (params[0] * b0 +
         params[1] * b1 +
         params[2] * b2 +
         params[3] * b3 +
         params[4] * b4 +
         params[5] * b5 +
         params[6] * b6 +
         params[7] * b7 +
         params[8] * b8 +
         params[9] * b9)
    
    return f
