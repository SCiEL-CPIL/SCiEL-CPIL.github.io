"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y using a 9th-degree Chebyshev polynomial basis on the domain [-2, 2].
    The input x is scaled to t in [-1, 1] to use the standard Chebyshev polynomials.
    y = sum_{k=0}^{9} params[k] * T_k(x/2)
    """
    # Scale x from [-2, 2] to [-1, 1]
    t = x / 2.0
    
    # Initialize Chebyshev polynomials T_0 and T_1
    T_k_minus_2 = np.ones_like(t)  # T_0(t) = 1
    T_k_minus_1 = t.copy()          # T_1(t) = t
    
    # Accumulate the weighted sum
    # We use params[0] for T_0 and params[1] for T_1
    y = params[0] * T_k_minus_2 + params[1] * T_k_minus_1
    
    # Compute T_k for k = 2 to 9 using the recurrence:
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    for k in range(2, 10):
        T_k = 2.0 * t * T_k_minus_1 - T_k_minus_2
        y += params[k] * T_k
        T_k_minus_2 = T_k_minus_1
        T_k_minus_1 = T_k
        
    return y
# EVOLVE-BLOCK-END
