"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial basis of degree 9 on [-3, 3].
    
    The input domain is [-3, 3]. We map x to t in [-1, 1] via t = x / 3.
    We then evaluate Chebyshev polynomials T_0(t), T_1(t), ..., T_9(t)
    and return a linear combination with coefficients params[0:10].
    
    This provides a well-conditioned, smooth, and expressive basis that can
    approximate a wide range of nonlinear functions.
    """
    # Map x from [-3, 3] to t in [-1, 1]
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0 through T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    T = np.zeros((10, len(x)))
    T[0, :] = 1.0
    T[1, :] = t
    
    for n in range(2, 10):
        T[n, :] = 2.0 * t * T[n-1, :] - T[n-2, :]
    
    # Return linear combination
    return T @ params
# EVOLVE-BLOCK-END
