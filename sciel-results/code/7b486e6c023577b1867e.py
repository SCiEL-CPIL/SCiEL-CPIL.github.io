"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-2, 2] to t in [-1, 1] for Chebyshev stability
    t = x / 2.0
    
    # Chebyshev polynomials of the first kind
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2 * t**2 - 1
    T3 = 4 * t**3 - 3 * t
    T4 = 8 * t**4 - 8 * t**2 + 1
    
    # Trigonometric basis
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2 * x)
    cos_2x = np.cos(2 * x)
    sin_3x = np.sin(3 * x)
    
    return (params[0] * T0 +
            params[1] * T1 +
            params[2] * T2 +
            params[3] * T3 +
            params[4] * T4 +
            params[5] * sin_x +
            params[6] * cos_x +
            params[7] * sin_2x +
            params[8] * cos_2x +
            params[9] * sin_3x)
# EVOLVE-BLOCK-END
