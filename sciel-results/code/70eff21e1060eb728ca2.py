import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-2, 2] to [-1, 1] for Chebyshev polynomials
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2 * t**2 - 1
    T3 = 4 * t**3 - 3 * t
    T4 = 8 * t**4 - 8 * t**2 + 1
    T5 = 16 * t**5 - 20 * t**3 + 5 * t
    T6 = 32 * t**6 - 48 * t**4 + 18 * t**2 - 1
    T7 = 64 * t**7 - 112 * t**5 + 56 * t**3 - 7 * t
    T8 = 128 * t**8 - 256 * t**6 + 160 * t**4 - 32 * t**2 + 1
    T9 = 256 * t**9 - 576 * t**7 + 432 * t**5 - 120 * t**3 + 9 * t
    
    # Linear combination
    y = (params[0] * T0 +
         params[1] * T1 +
         params[2] * T2 +
         params[3] * T3 +
         params[4] * T4 +
         params[5] * T5 +
         params[6] * T6 +
         params[7] * T7 +
         params[8] * T8 +
         params[9] * T9)
         
    return y
