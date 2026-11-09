"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev series of degree 9 (10 coefficients).
    Maps x from [0.1, 6] to [-1, 1] and evaluates T_0..T_9.
    """
    # Map x from [0.1, 6] to [-1, 1]
    # t = (2*x - (0.1 + 6)) / (6 - 0.1) = (2*x - 6.1) / 5.9
    t = (2.0 * x - 6.1) / 5.9
    
    # Clamp to avoid numerical issues slightly outside [-1, 1]
    t = np.clip(t, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_0 through T_9 using recurrence
    n = 10
    T = np.zeros((n, len(x)))
    T[0, :] = 1.0
    T[1, :] = t
    for k in range(2, n):
        T[k, :] = 2.0 * t * T[k-1, :] - T[k-2, :]
    
    # T is (10, N), params is (10,)
    # Result: sum over k of params[k] * T[k, i]
    result = T.T @ params
    
    return result
# EVOLVE-BLOCK-END
