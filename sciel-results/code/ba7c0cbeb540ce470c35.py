"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-2, 2] to t in [-1, 1] for Chebyshev stability
    t = x / 2.0
    
    # Chebyshev polynomials of the first kind (computed via recurrence for stability)
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2 * t**2 - 1
    T3 = 4 * t**3 - 3 * t
    T4 = 8 * t**4 - 8 * t**2 + 1
    T5 = 16 * t**5 - 20 * t**3 + 5 * t
    
    # Trigonometric basis
    # Using frequencies adapted to the domain. 
    # sin(x), cos(x) cover the base frequency.
    # sin(2x), cos(2x) cover the second harmonic.
    # tanh(x) provides a smooth, bounded non-linear feature that can capture saturation.
    
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2 * x)
    cos_2x = np.cos(2 * x)
    tanh_x = np.tanh(x)
    
    # Combine features
    # We use 10 parameters as required.
    # 0: bias (T0)
    # 1: T1 (linear in t)
    # 2: T2 (quadratic-ish)
    # 3: T3 (cubic-ish)
    # 4: T4 (quartic-ish)
    # 5: sin(x)
    # 6: cos(x)
    # 7: sin(2x)
    # 8: cos(2x)
    # 9: tanh(x)
    
    result = (params[0] * T0 +
              params[1] * T1 +
              params[2] * T2 +
              params[3] * T3 +
              params[4] * T4 +
              params[5] * sin_x +
              params[6] * cos_x +
              params[7] * sin_2x +
              params[8] * cos_2x +
              params[9] * tanh_x)
              
    return result
# EVOLVE-BLOCK-END
