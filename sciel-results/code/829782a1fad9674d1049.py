"""Rewritten program for symbolic regression with 1 input."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Chebyshev polynomial basis of degree 9.
    
    The model is a linear combination of 10 Chebyshev polynomials T_0(x) through T_9(x).
    Chebyshev polynomials are well-conditioned on [-1, 1], and since x is in [-2, 2],
    we scale to [-1, 1] for numerical stability. This provides a smooth, well-behaved
    landscape for BFGS optimization while having the expressive power of a 9th-degree
    polynomial to capture a wide range of functions.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Scale x from [-2, 2] to [-1, 1] for Chebyshev stability
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 through T_9 using the recurrence relation
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n}(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    T = np.zeros((10, len(t)), dtype=np.float64)
    T[0, :] = 1.0
    if len(t) > 0:
        T[1, :] = t
    for n in range(2, 10):
        T[n, :] = 2.0 * t * T[n-1, :] - T[n-2, :]
    
    # Linear combination: sum_{k=0}^{9} p[k] * T_k(t)
    result = T @ p
    
    return result

# EVOLVE-BLOCK-END
