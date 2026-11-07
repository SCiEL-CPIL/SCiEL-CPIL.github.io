"""Symbolic regression using a Chebyshev polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x)
    
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Map x from [-3, 3] to t in [-1, 1]
    t = x / 3.0
    
    # Clip t to [-1, 1] to avoid numerical issues with arccos if x slightly exceeds bounds
    # though the problem states x is drawn from [-3, 3], clipping is safe.
    t = np.clip(t, -1.0, 1.0)
    
    # Compute Chebyshev polynomials of the first kind T_0 to T_9
    # T_n(t) = cos(n * arccos(t))
    # This is stable and efficient for n up to 9.
    
    # Precompute arccos(t) once
    theta = np.arccos(t)
    
    # Generate basis functions
    # T_0 = 1
    # T_1 = cos(theta) = t
    # T_2 = cos(2*theta)
    # ...
    # T_9 = cos(9*theta)
    
    y = np.zeros_like(x)
    
    # Unroll for speed and clarity, using the 10 parameters
    # p[0] * T_0
    # p[1] * T_1
    # ...
    # p[9] * T_9
    
    # Using a loop is fast enough for n=10, but unrolling avoids loop overhead.
    # However, np.cos(k * theta) is vectorized.
    
    # Let's build the basis matrix implicitly in the sum
    # y = sum_{k=0}^9 p[k] * cos(k * theta)
    
    # We can do this efficiently:
    # k = np.arange(10)
    # basis = np.cos(k[:, None] * theta[None, :])
    # y = basis @ p
    
    # This creates a (10, N) array and does a matrix-vector product.
    # For N=200, this is very fast.
    
    k = np.arange(10, dtype=np.float64)
    # theta is shape (N,)
    # k[:, None] is shape (10, 1)
    # theta[None, :] is shape (1, N)
    # k[:, None] * theta[None, :] is shape (10, N)
    
    basis = np.cos(k[:, None] * theta[None, :])
    
    # p is shape (10,)
    # y = basis @ p -> shape (N,)
    y = basis @ p
    
    return y
# EVOLVE-BLOCK-END
