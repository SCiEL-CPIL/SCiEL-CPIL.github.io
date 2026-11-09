"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize input to [-1, 1] for stable Chebyshev evaluation
    t = x / 3.0
    
    # Chebyshev polynomials of the first kind T_n(t)
    # T_0 = 1
    # T_1 = t
    # T_{n} = 2t T_{n-1} - T_{n-2}
    
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2 * t * T1 - T0
    T3 = 2 * t * T2 - T1
    T4 = 2 * t * T3 - T2
    T5 = 2 * t * T4 - T3
    T6 = 2 * t * T5 - T4
    T7 = 2 * t * T6 - T5
    T8 = 2 * t * T7 - T6
    T9 = 2 * t * T8 - T7
    
    # Combine with parameters, using decaying scale for higher-order terms
    # to improve BFGS convergence from the all-ones initialization
    scale = np.array([1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.1, 0.1, 0.05, 0.05])
    y = (params[0] * scale[0] * T0 +
         params[1] * scale[1] * T1 +
         params[2] * scale[2] * T2 +
         params[3] * scale[3] * T3 +
         params[4] * scale[4] * T4 +
         params[5] * scale[5] * T5 +
         params[6] * scale[6] * T6 +
         params[7] * scale[7] * T7 +
         params[8] * scale[8] * T8 +
         params[9] * scale[9] * T9)
    
    return y
# EVOLVE-BLOCK-END
