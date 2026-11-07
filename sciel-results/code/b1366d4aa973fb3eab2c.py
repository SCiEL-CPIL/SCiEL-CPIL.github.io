"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev Series Basis Function.
    
    Uses 10 Chebyshev polynomials of the first kind mapped to [-1, 1].
    This provides a highly stable linear basis for polynomial and smooth functions.
    Parameters p[0..9] are the coefficients for T_0 to T_9.
    """
    p = params
    
    # Safety check for finite params
    if not np.all(np.isfinite(p)):
        p = np.ones(10)
        
    # Map x from [-2, 2] to [-1, 1] for standard Chebyshev domain
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0 = 1
    # T_1 = t
    # T_n = 2*t*T_{n-1} - T_{n-2}
    
    T = np.zeros((10, len(x)))
    T[0, :] = 1.0
    if len(x) > 0:
        T[1, :] = t
        
        for n in range(2, 10):
            T[n, :] = 2.0 * t * T[n-1, :] - T[n-2, :]
            
    # Linear combination
    # result = sum(p[i] * T[i])
    # Using matrix multiplication for efficiency
    result = T.T @ p
    
    # Safety: replace any non-finite values with 0
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
