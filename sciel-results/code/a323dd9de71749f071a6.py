import numpy as np
import time

_START_TIME = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial basis on mapped domain [0.1, 6].
    
    Maps x from [0.1, 6] to t in [-1, 1] for optimal Chebyshev approximation.
    Uses 10 Chebyshev polynomials T_0 through T_9, which are orthogonal
    on [-1, 1] with respect to the weight 1/sqrt(1-t^2), providing a
    well-conditioned basis for smooth function approximation.
    
    The model is linear in params, so BFGS converges to the global optimum
    of this fixed basis in very few iterations.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map x from [0.1, 6] to t in [-1, 1]
    # t = (2*x - (x_min + x_max)) / (x_max - x_min)
    x_min, x_max = 0.1, 6.0
    t = (2.0 * x - (x_min + x_max)) / (x_max - x_min)
    
    # Compute Chebyshev polynomials T_0 through T_9 using recurrence
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    T0 = np.ones_like(t)
    T1 = t.copy()
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    T5 = 2.0 * t * T4 - T3
    T6 = 2.0 * t * T5 - T4
    T7 = 2.0 * t * T6 - T5
    T8 = 2.0 * t * T7 - T6
    T9 = 2.0 * t * T8 - T7
    
    y = (params[0] * T0
         + params[1] * T1
         + params[2] * T2
         + params[3] * T3
         + params[4] * T4
         + params[5] * T5
         + params[6] * T6
         + params[7] * T7
         + params[8] * T8
         + params[9] * T9)
    
    return y
