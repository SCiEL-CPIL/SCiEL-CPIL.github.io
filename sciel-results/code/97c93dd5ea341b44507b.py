"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial basis of degree 9 on the scaled interval [-1, 1].
    
    x is in [0.1, 6]. We scale to t = (x - 3.05) / 2.95 to map to [-1, 1].
    Then we use Chebyshev polynomials T_0 through T_9.
    
    This provides:
    - Near-orthogonality on the interval (with respect to weight 1/sqrt(1-t^2))
    - Better numerical conditioning than monomials
    - Ability to represent smooth functions up to degree 9
    - Stability for BFGS optimization
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure finite values
    x = np.where(np.isfinite(x), x, 3.05)
    p = np.where(np.isfinite(p), p, 0.0)
    
    # Scale x from [0.1, 6] to [-1, 1]
    # Midpoint = (0.1 + 6) / 2 = 3.05
    # Half-width = (6 - 0.1) / 2 = 2.95
    t = (x - 3.05) / 2.95
    
    # Clip to [-1, 1] to avoid numerical issues (though x should be in [0.1, 6])
    t = np.clip(t, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_0 through T_9 using recurrence
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n+1}(t) = 2*t*T_n(t) - T_{n-1}(t)
    
    T = np.zeros((10, x.shape[0]))
    T[0, :] = 1.0
    T[1, :] = t
    
    for n in range(2, 10):
        T[n, :] = 2.0 * t * T[n-1, :] - T[n-2, :]
    
    # Combine with parameters
    # prediction = sum_{k=0}^{9} p[k] * T_k(t)
    result = np.zeros(x.shape[0])
    for k in range(10):
        result += p[k] * T[k, :]
    
    return result
# EVOLVE-BLOCK-END
