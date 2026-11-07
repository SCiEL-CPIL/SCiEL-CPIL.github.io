# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate an unknown function y = f(x) for x in [-2, 2] using a linear
    combination of 10 well-conditioned basis functions.
    
    Basis: Chebyshev polynomials T_0 through T_9 on [-1, 1] (mapped from [-2, 2]).
    These are orthogonal on [-1, 1] and well-conditioned, making them ideal for
    least-squares fitting.
    """
    # Guard against non-finite params
    if not np.all(np.isfinite(params)):
        params = np.ones(10)
    
    # Ensure x is a 1-D array
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    
    if n == 0:
        return np.array([])
    
    # Map x from [-2, 2] to [-1, 1] for Chebyshev stability
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    T = np.empty((n, 10), dtype=np.float64)
    T[:, 0] = 1.0
    T[:, 1] = t
    
    for k in range(2, 10):
        T[:, k] = 2.0 * t * T[:, k-1] - T[:, k-2]
    
    # Return the linear combination
    return T @ params
# EVOLVE-BLOCK-END
