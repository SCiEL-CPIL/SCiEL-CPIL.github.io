"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Polynomial basis using Chebyshev polynomials of the first kind.
    Basis: [T_0(x/2), T_1(x/2), ..., T_9(x/2)]
    This provides a well-conditioned basis for fitting smooth functions
    in the range [-2, 2], improving numerical stability for BFGS.
    """
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=float)
    
    # Scale input to [-1, 1] for standard Chebyshev polynomials
    t = x / 2.0
    
    # Extract parameters
    p = params
    
    # Compute Chebyshev polynomials T_0 to T_9 using recurrence
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n+1}(t) = 2*t*T_n(t) - T_{n-1}(t)
    
    T_0 = np.ones_like(t)
    T_1 = t
    
    # Recursively compute higher order terms
    T_2 = 2.0 * t * T_1 - T_0
    T_3 = 2.0 * t * T_2 - T_1
    T_4 = 2.0 * t * T_3 - T_2
    T_5 = 2.0 * t * T_4 - T_3
    T_6 = 2.0 * t * T_5 - T_4
    T_7 = 2.0 * t * T_6 - T_5
    T_8 = 2.0 * t * T_7 - T_6
    T_9 = 2.0 * t * T_8 - T_7
    
    # Combine with parameters
    result = (p[0] * T_0 + p[1] * T_1 + p[2] * T_2 + p[3] * T_3 + 
              p[4] * T_4 + p[5] * T_5 + p[6] * T_6 + p[7] * T_7 + 
              p[8] * T_8 + p[9] * T_9)
    
    return result
# EVOLVE-BLOCK-END
