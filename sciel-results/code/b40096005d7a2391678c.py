"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.ravel()
    
    # Scale x from [-3,3] to [-1,1] for Legendre polynomials
    t = x / 3.0
    
    # Legendre polynomials P_0 through P_9 evaluated at t
    P0 = np.ones_like(t)
    P1 = t
    P2 = (3.0 * t**2 - 1.0) / 2.0
    P3 = (5.0 * t**3 - 3.0 * t) / 2.0
    P4 = (35.0 * t**4 - 30.0 * t**2 + 3.0) / 8.0
    P5 = (63.0 * t**5 - 70.0 * t**3 + 15.0 * t) / 8.0
    P6 = (231.0 * t**6 - 315.0 * t**4 + 105.0 * t**2 - 5.0) / 16.0
    P7 = (429.0 * t**7 - 693.0 * t**5 + 315.0 * t**3 - 35.0 * t) / 16.0
    P8 = (6435.0 * t**8 - 12012.0 * t**6 + 6930.0 * t**4 - 1260.0 * t**2 + 35.0) / 128.0
    P9 = (12155.0 * t**9 - 25740.0 * t**7 + 20790.0 * t**5 - 6006.0 * t**3 + 429.0 * t) / 128.0
    
    # Build prediction: sum of param_i * Legendre_i
    pred = (params[0] * P0 +
            params[1] * P1 +
            params[2] * P2 +
            params[3] * P3 +
            params[4] * P4 +
            params[5] * P5 +
            params[6] * P6 +
            params[7] * P7 +
            params[8] * P8 +
            params[9] * P9)
    
    return pred
# EVOLVE-BLOCK-END
