"""Fourier series basis with linear drift term for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series basis (frequencies 1-4) + linear drift term.
    
    params[0]   -> constant term
    params[1]   -> sin(x)
    params[2]   -> cos(x)
    params[3]   -> sin(2x)
    params[4]   -> cos(2x)
    params[5]   -> sin(3x)
    params[6]   -> cos(3x)
    params[7]   -> sin(4x)
    params[8]   -> cos(4x)
    params[9]   -> linear term x (for non-periodic drift)
    
    This gives a well-conditioned, orthogonal-ish basis on [-3, 3] that can
    capture periodic features up to frequency 4, plus a linear trend.
    """
    result = np.zeros_like(x, dtype=np.float64)
    
    # Constant term
    result += params[0]
    
    # Fourier terms for k = 1, 2, 3, 4
    # sin(k*x) terms at indices 1, 3, 5, 7
    # cos(k*x) terms at indices 2, 4, 6, 8
    result += params[1] * np.sin(x)
    result += params[2] * np.cos(x)
    result += params[3] * np.sin(2 * x)
    result += params[4] * np.cos(2 * x)
    result += params[5] * np.sin(3 * x)
    result += params[6] * np.cos(3 * x)
    result += params[7] * np.sin(4 * x)
    result += params[8] * np.cos(4 * x)
    
    # Linear drift term
    result += params[9] * x
    
    return result
# EVOLVE-BLOCK-END
