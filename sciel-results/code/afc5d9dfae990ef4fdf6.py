"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structural change: use a Chebyshev polynomial basis of degree 9 (all 10 params).
    This is a linear model in params, so BFGS will converge globally (the problem is
    convex). Chebyshev polynomials are well-conditioned on [-1, 1] and form a complete
    basis for continuous functions on that interval, giving strong approximation power
    for smooth unknown functions without the non-convexity issues of learnable
    frequencies.
    
    Parameters (length 10):
    p0..p9: coefficients for T_0(z), T_1(z), ..., T_9(z) where z = x/2 maps [-2,2] to [-1,1]
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Scale to [-1, 1] for Chebyshev stability
    z = x / 2.0
    
    # Compute Chebyshev polynomials T_0 through T_9 using recurrence
    # T_0(z) = 1
    # T_1(z) = z
    # T_{n+1}(z) = 2*z*T_n(z) - T_{n-1}(z)
    T = np.zeros((10, x.size), dtype=np.float64)
    T[0, :] = 1.0
    T[1, :] = z
    for n in range(2, 10):
        T[n, :] = 2.0 * z * T[n-1, :] - T[n-2, :]
    
    # Linear combination: y = sum_{k=0}^{9} p_k * T_k(z)
    y = T @ params
    
    return y
# EVOLVE-BLOCK-END
