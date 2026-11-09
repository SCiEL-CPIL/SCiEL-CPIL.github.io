"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale input from [-3, 3] to [-1, 1] for Chebyshev polynomials
    t = x / 3.0
    
    # Precompute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    T0 = np.ones_like(t)
    T1 = t
    
    # We need T_0 ... T_9
    # params[0] * T0 + params[1] * T1 + ... + params[9] * T9
    
    if len(params) < 2:
        # Fallback if params are shorter than expected, though contract says length 10
        return params[0] * T0
    
    # Build the basis matrix or compute term by term to save memory
    # Term by term is efficient for small N (10 terms)
    
    result = params[0] * T0 + params[1] * T1
    
    if len(params) > 2:
        T_prev2 = T0
        T_prev1 = T1
        for k in range(2, 10):
            if k >= len(params):
                break
            # Recurrence: T_k = 2*t*T_{k-1} - T_{k-2}
            T_curr = 2.0 * t * T_prev1 - T_prev2
            result += params[k] * T_curr
            T_prev2 = T_prev1
            T_prev1 = T_curr
            
    return result
# EVOLVE-BLOCK-END
