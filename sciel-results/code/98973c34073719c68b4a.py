import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with a separable tensor-product Chebyshev basis.
    
    Uses T_i(x) * T_j(y) for i,j in {0,1,2} (9 terms) plus T3(x)*T0(y)
    as the 10th term. This complete tensor-product structure is well-conditioned
    on [-1,1]^2 and allows BFGS to converge reliably.
    
    Basis:
    0: T0(x)*T0(y) = 1
    1: T1(x)*T0(y) = x
    2: T2(x)*T0(y) = 2x^2-1
    3: T0(x)*T1(y) = y
    4: T1(x)*T1(y) = x*y
    5: T2(x)*T1(y) = (2x^2-1)*y
    6: T0(x)*T2(y) = 2y^2-1
    7: T1(x)*T2(y) = x*(2y^2-1)
    8: T2(x)*T2(y) = (2x^2-1)*(2y^2-1)
    9: T3(x)*T0(y) = 4x^3-3x
    """
    # Chebyshev Polynomials of the First Kind
    T0 = 1.0
    T1_x = x
    T2_x = 2.0 * x * x - 1.0
    T3_x = 4.0 * x * x * x - 3.0 * x
    
    T1_y = y
    T2_y = 2.0 * y * y - 1.0
    
    # Tensor product basis: T_i(x) * T_j(y) for i,j in {0,1,2}
    f = (params[0] * T0 * T0 +
         params[1] * T1_x * T0 +
         params[2] * T2_x * T0 +
         params[3] * T0 * T1_y +
         params[4] * T1_x * T1_y +
         params[5] * T2_x * T1_y +
         params[6] * T0 * T2_y +
         params[7] * T1_x * T2_y +
         params[8] * T2_x * T2_y +
         params[9] * T3_x * T0)
         
    return f
