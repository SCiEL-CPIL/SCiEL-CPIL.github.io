import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    # Chebyshev polynomials of the first kind on [-1, 1]
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2t^2 - 1
    # T3(t) = 4t^3 - 3t
    
    # Precompute Chebyshev polynomials for x (degrees 0-3)
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x * x - 1.0
    T3_x = 4.0 * x**3 - 3.0 * x
    
    # Precompute Chebyshev polynomials for y (degrees 0-2)
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y * y - 1.0
    
    # Construct 10 basis functions from tensor products
    # Ordered by total degree to capture low-frequency components first
    term0 = T0_x * T0_y  # 1
    term1 = T1_x * T0_y  # x
    term2 = T0_x * T1_y  # y
    term3 = T2_x * T0_y  # 2x^2 - 1
    term4 = T0_x * T2_y  # 2y^2 - 1
    term5 = T1_x * T1_y  # xy
    term6 = T3_x * T0_y  # 4x^3 - 3x
    term7 = T2_x * T1_y  # (2x^2-1)y
    term8 = T1_x * T2_y  # x(2y^2-1)
    term9 = T2_x * T2_y  # (2x^2-1)(2y^2-1)

    return (p[0] * term0 +
            p[1] * term1 +
            p[2] * term2 +
            p[3] * term3 +
            p[4] * term4 +
            p[5] * term5 +
            p[6] * term6 +
            p[7] * term7 +
            p[8] * term8 +
            p[9] * term9)
