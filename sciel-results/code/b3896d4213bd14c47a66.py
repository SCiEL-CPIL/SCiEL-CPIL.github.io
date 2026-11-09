"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate a hidden function using shifted Chebyshev polynomials of the first kind.
    These are orthogonal with respect to the uniform distribution on [-1, 1] and
    provide excellent conditioning and spectral convergence for smooth functions.
    
    Parameters:
    x : np.ndarray
        Input points.
    params : np.ndarray
        Array of 10 parameters to be fitted.
        
    Returns:
    np.ndarray
        Predicted values.
    """
    # Scale x from [-3, 3] to [-1, 1]
    u = x / 3.0
    
    # Compute shifted Chebyshev polynomials T_0(u) to T_9(u)
    # Using the recurrence relation: T_0(t) = 1, T_1(t) = t, T_{n+1}(t) = 2*t*T_n(t) - T_{n-1}(t)
    # This is more efficient and numerically stable than using cos(n * arccos(t))
    
    # Precompute the Chebyshev polynomial values
    V = np.zeros((len(x), 10))
    V[:, 0] = 1.0
    if len(params) > 1:
        V[:, 1] = u
        for k in range(2, 10):
            V[:, k] = 2.0 * u * V[:, k-1] - V[:, k-2]
    
    # Linear combination of basis functions
    return V @ params
# EVOLVE-BLOCK-END
