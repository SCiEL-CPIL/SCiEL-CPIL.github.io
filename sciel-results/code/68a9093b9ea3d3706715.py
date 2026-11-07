"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros_like(x)

    n = len(x)
    if n == 0:
        return np.array([])

    # Map x to [-1, 1] for Chebyshev
    t = x / 2.0

    # Compute Chebyshev polynomials T_0 through T_9 using recurrence
    T = np.empty((10, n))
    T[0] = 1.0
    T[1] = t
    for k in range(2, 10):
        T[k] = 2.0 * t * T[k-1] - T[k-2]

    # params[0..9] are coefficients for T_0..T_9
    result = np.zeros(n)
    for k in range(10):
        result += params[k] * T[k]

    return result
# EVOLVE-BLOCK-END
