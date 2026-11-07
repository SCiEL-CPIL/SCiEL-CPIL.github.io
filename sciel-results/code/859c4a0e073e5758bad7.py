"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Chebyshev polynomials (degree 0-9) on mapped domain [-1,1].
    
    This uses a well-conditioned orthogonal basis that should allow BFGS to
    converge reliably. The Chebyshev basis is particularly good for approximating
    smooth functions on a bounded interval.
    
    params[0..9] are coefficients for T_0 through T_9.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Map x from [-2, 2] to [-1, 1]
    u = 0.5 * x
    
    # Compute Chebyshev polynomials T_0 through T_9 using recurrence
    n = len(x)
    T = np.zeros((10, n), dtype=np.float64)
    T[0, :] = 1.0
    if n > 0:
        T[1, :] = u
    for k in range(2, 10):
        T[k, :] = 2.0 * u * T[k-1, :] - T[k-2, :]
    
    # Compute result as linear combination
    result = T.T @ params  # shape (n,)
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
