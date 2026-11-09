"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the target function using a Chebyshev polynomial basis of degree 9.
    This model is linear in the parameters, ensuring robust convergence for BFGS.
    The Chebyshev basis is well-conditioned on [-2, 2] and can represent a wide
    range of smooth functions, potentially outperforming the linear baseline.
    """
    # Map x from [-2, 2] to t in [-1, 1] to use standard Chebyshev polynomials
    t = x / 2.0
    
    # Precompute Chebyshev polynomials T_0 through T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    T = np.zeros((10, x.size))
    T[0, :] = 1.0
    T[1, :] = t
    
    for k in range(2, 10):
        T[k, :] = 2.0 * t * T[k-1, :] - T[k-2, :]
        
    # The model is a linear combination of these basis functions
    return T @ params
# EVOLVE-BLOCK-END
