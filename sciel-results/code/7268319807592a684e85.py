"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # Safety margin before 150s kill


def _elapsed_ok() -> bool:
    return (time.time() - _start_time) < _DEADLINE


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y using a 10-term Fourier series with fixed frequencies {1, 2, 3, 4, 5}
    (sines and cosines) as a linear-in-params basis.
    x: input array
    params: array of 10 coefficients
    Returns: predictions array same shape as x
    """
    if not _elapsed_ok():
        return np.zeros_like(x)
    
    # Fourier series: sum of a_k * cos(k*x) + b_k * sin(k*x) for k = 1..5
    # params[0..4] = cosine coefficients for k=1..5
    # params[5..9] = sine coefficients for k=1..5
    
    y = np.zeros_like(x)
    
    for k in range(1, 6):
        y += params[k - 1] * np.cos(k * x)
        y += params[k + 4] * np.sin(k * x)
    
    return y
# EVOLVE-BLOCK-END
