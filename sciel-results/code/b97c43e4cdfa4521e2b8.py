import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Use Chebyshev polynomial basis T_i(x) * T_j(y) for total degree <= 3
    # This provides better numerical conditioning than monomials while
    # spanning the same function space.
    #
    # Chebyshev polynomials:
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    #
    # Basis functions (total degree i+j <= 3):
    # 0: T_0(x)T_0(y) = 1
    # 1: T_1(x)T_0(y) = x
    # 2: T_0(x)T_1(y) = y
    # 3: T_2(x)T_0(y) = 2x^2 - 1
    # 4: T_0(x)T_2(y) = 2y^2 - 1
    # 5: T_1(x)T_1(y) = x*y
    # 6: T_3(x)T_0(y) = 4x^3 - 3x
    # 7: T_0(x)T_3(y) = 4y^3 - 3y
    # 8: T_2(x)T_1(y) = (2x^2-1)*y
    # 9: T_1(x)T_2(y) = x*(2y^2-1)
    
    # Compute Chebyshev polynomials
    T0 = np.ones_like(x)
    T1x = x
    T1y = y
    T2x = 2.0 * x * x - 1.0
    T2y = 2.0 * y * y - 1.0
    T3x = 4.0 * x * x * x - 3.0 * x
    T3y = 4.0 * y * y * y - 3.0 * y
    
    # Vectorized computation using Chebyshev basis
    pred = (params[0] * T0 * T0 + 
            params[1] * T1x * T0 + 
            params[2] * T0 * T1y + 
            params[3] * T2x * T0 + 
            params[4] * T0 * T2y + 
            params[5] * T1x * T1y + 
            params[6] * T3x * T0 + 
            params[7] * T0 * T3y + 
            params[8] * T2x * T1y + 
            params[9] * T1x * T2y)
            
    return pred
