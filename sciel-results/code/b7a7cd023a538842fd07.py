"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a Chebyshev polynomial basis of degree 9.
    The input x is mapped from [-2, 2] to [-1, 1] to utilize standard Chebyshev polynomials.
    This basis is orthogonal and well-conditioned, facilitating stable convergence for BFGS.
    """
    # Map x from [-2, 2] to [-1, 1]
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0(t) through T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    T = np.empty((10, len(x)))
    T[0] = 1.0
    T[1] = t
    
    for k in range(2, 10):
        T[k] = 2.0 * t * T[k-1] - T[k-2]
        
    # Linear combination of the basis functions
    # params[i] corresponds to the coefficient for T_i(t)
    prediction = np.sum(params[:, np.newaxis] * T, axis=0)
    
    return prediction
# EVOLVE-BLOCK-END
