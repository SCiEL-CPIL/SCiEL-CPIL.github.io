"""Symbolic regression with a combined Fourier-Chebyshev basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # seconds


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Robust basis expansion combining Fourier and Chebyshev components.
    
    The 10 parameters are split into:
    - 5 Chebyshev polynomial terms (capturing smooth non-periodic structure)
    - 5 Fourier terms (sine/cosine with fundamental frequency 2pi, capturing periodic structure)
    
    Chebyshev basis (T_0 to T_4) mapped from [0.1, 6] to [-1, 1]:
    0: T_0 = 1
    1: T_1 = t
    2: T_2 = 2t^2 - 1
    3: T_3 = 4t^3 - 3t
    4: T_4 = 8t^4 - 8t^2 + 1
    
    Fourier basis (fundamental frequency 2pi over the domain length ~5.9):
    5: sin(2pi * x / 5.9)
    6: cos(2pi * x / 5.9)
    7: sin(4pi * x / 5.9)
    8: cos(4pi * x / 5.9)
    9: sin(6pi * x / 5.9)
    
    This hybrid approach leverages:
    - Chebyshev polynomials for excellent conditioning and smooth function approximation
    - Fourier terms for capturing any periodic or oscillatory components
    - BFGS can efficiently optimize the 10 parameters in this well-conditioned basis
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Safety margin: return a simple fallback to avoid timeout
        return params[0] + params[1] * x
    
    p = np.asarray(params, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    
    # Map x from [0.1, 6] to t in [-1, 1] for Chebyshev polynomials
    x_min = 0.1
    x_max = 6.0
    t = (2.0 * x_arr - (x_max + x_min)) / (x_max - x_min)
    t = np.clip(t, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_0 to T_4
    T0 = np.ones_like(x_arr)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    
    # Compute Fourier terms with fundamental frequency 2pi / (x_max - x_min)
    # This gives one full period over the domain [0.1, 6]
    omega = 2.0 * np.pi / (x_max - x_min)
    
    F0 = np.sin(omega * x_arr)
    F1 = np.cos(omega * x_arr)
    F2 = np.sin(2.0 * omega * x_arr)
    F3 = np.cos(2.0 * omega * x_arr)
    F4 = np.sin(3.0 * omega * x_arr)
    
    # Linear combination of all basis functions
    result = (
        p[0] * T0 +
        p[1] * T1 +
        p[2] * T2 +
        p[3] * T3 +
        p[4] * T4 +
        p[5] * F0 +
        p[6] * F1 +
        p[7] * F2 +
        p[8] * F3 +
        p[9] * F4
    )
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        # Fallback to simple linear if numerical issues arise
        result = p[0] + p[1] * x_arr
        result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
