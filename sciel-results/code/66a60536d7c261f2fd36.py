"""Symbolic regression with Chebyshev polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear model using Chebyshev polynomials of degree 0-9.
    
    Chebyshev polynomials T_k(t) are defined on [-1,1] and have the
    recurrence: T_0=1, T_1=t, T_k = 2*t*T_{k-1} - T_{k-2}.
    
    We map x from [-3,3] to [-1,1] for numerical stability.
    The model is linear in params, so BFGS converges to the global
    least-squares optimum reliably.
    
    params: [c0, c1, c2, ..., c9] - coefficients for T_0 through T_9
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map x from [-3, 3] to [-1, 1] for Chebyshev evaluation
    t = x / 3.0
    t = np.clip(t, -1.0, 1.0)  # safety
    
    # Build Chebyshev basis via recurrence
    T = np.zeros((10, len(x)), dtype=np.float64)
    T[0, :] = 1.0
    if len(x) > 0:
        T[1, :] = t
    for k in range(2, 10):
        T[k, :] = 2.0 * t * T[k-1, :] - T[k-2, :]
    
    # Linear combination: y = sum(c_k * T_k)
    # params shape: (10,), T shape: (10, N)
    y = params @ T  # result shape: (N,)
    
    # Ensure finite output
    y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return y
# EVOLVE-BLOCK-END
