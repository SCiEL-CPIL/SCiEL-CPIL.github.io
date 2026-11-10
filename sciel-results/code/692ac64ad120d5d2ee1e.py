"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Use a Chebyshev polynomial basis of degree 9 on [-3, 3].
    Map x from [-3, 3] to [-1, 1] for standard Chebyshev polynomials.
    Use 10 parameters as coefficients for T_0 through T_9.
    """
    # Map x from [-3, 3] to [-1, 1]
    t = x / 3.0
    
    # Build Chebyshev polynomial values T_0(t) through T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n+1}(t) = 2*t*T_n(t) - T_{n-1}(t)
    
    T = np.zeros((10, len(t)))
    T[0, :] = 1.0
    if len(t) > 0:
        T[1, :] = t
        for n in range(2, 10):
            T[n, :] = 2.0 * t * T[n-1, :] - T[n-2, :]
    
    # params is a 1D array of length 10
    coeffs = params[:10]
    
    # Compute the weighted sum of Chebyshev polynomials
    prediction = T @ coeffs
    
    return prediction
# EVOLVE-BLOCK-END
