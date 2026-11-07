import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using an orthogonal basis of Chebyshev polynomials
    up to degree 3 in each variable.
    
    Chebyshev polynomials are optimal for approximating smooth functions on [-1, 1]
    as they minimize the maximum error (minimax property) and have better
    convergence properties than monomials or Legendre polynomials for non-polynomial
    targets. The basis spans the space of polynomials of total degree <= 3.
    
    Basis terms:
    0: T0(x)T0(y) = 1
    1: T1(x)T0(y) = x
    2: T0(x)T1(y) = y
    3: T2(x)T0(y) = 2x^2 - 1
    4: T0(x)T2(y) = 2y^2 - 1
    5: T1(x)T1(y) = x * y
    6: T3(x)T0(y) = 4x^3 - 3x
    7: T0(x)T3(y) = 4y^3 - 3y
    8: T2(x)T1(y) = (2x^2 - 1) * y
    9: T1(x)T2(y) = x * (2y^2 - 1)
    """
    # Chebyshev polynomials of the first kind
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2t^2 - 1
    # T3(t) = 4t^3 - 3t
    
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2 * x**2 - 1
    T3_x = 4 * x**3 - 3 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2 * y**2 - 1
    T3_y = 4 * y**3 - 3 * y
    
    # Construct the 10 basis functions
    b0 = T0_x * T0_y
    b1 = T1_x * T0_y
    b2 = T0_x * T1_y
    b3 = T2_x * T0_y
    b4 = T0_x * T2_y
    b5 = T1_x * T1_y
    b6 = T3_x * T0_y
    b7 = T0_x * T3_y
    b8 = T2_x * T1_y
    b9 = T1_x * T2_y
    
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
