import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Chebyshev polynomial basis.
    
    Strategy:
    - Uses Chebyshev polynomials of the first kind, which are orthogonal on [-1, 1].
    - This provides superior numerical conditioning compared to monomial bases or
      local RBFs, helping BFGS converge more reliably and accurately.
    - The basis consists of product terms T_i(x) * T_j(y).
    - We select 10 terms to cover a reasonable range of low-to-mid frequency features:
      (0,0), (1,0), (0,1), (2,0), (0,2), (1,1), (3,0), (0,3), (2,1), (1,2)
    - This selection balances pure x/y dependencies and interactions.
    - BFGS fits the 10 coefficients.
    """
    start_time = time.time()
    
    p = params
    
    # Pre-compute Chebyshev polynomials T_0, T_1, T_2, T_3 for x and y
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    
    # T_0
    T0x = np.ones_like(x)
    T0y = np.ones_like(y)
    
    # T_1
    T1x = x
    T1y = y
    
    # T_2
    T2x = 2.0 * x**2 - 1.0
    T2y = 2.0 * y**2 - 1.0
    
    # T_3
    T3x = 4.0 * x**3 - 3.0 * x
    T3y = 4.0 * y**3 - 3.0 * y
    
    # Construct the 10 basis functions
    # 1: T0(x)*T0(y)
    b0 = T0x * T0y
    # 2: T1(x)*T0(y)
    b1 = T1x * T0y
    # 3: T0(x)*T1(y)
    b2 = T0x * T1y
    # 4: T2(x)*T0(y)
    b3 = T2x * T0y
    # 5: T0(x)*T2(y)
    b4 = T0x * T2y
    # 6: T1(x)*T1(y)
    b5 = T1x * T1y
    # 7: T3(x)*T0(y)
    b6 = T3x * T0y
    # 8: T0(x)*T3(y)
    b7 = T0x * T3y
    # 9: T2(x)*T1(y)
    b8 = T2x * T1y
    # 10: T1(x)*T2(y)
    b9 = T1x * T2y
    
    # Linear combination
    pred = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + 
            p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 + 
            p[8] * b8 + p[9] * b9)
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.zeros_like(x)
        
    # Deadline guard
    if time.time() - start_time > 110:
        return np.zeros_like(x)
        
    return pred
