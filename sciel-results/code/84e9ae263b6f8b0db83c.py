"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return np.array([])
    
    # Nonlinear pre-transform: map [-2,2] to [-1,1] with sin for non-polynomial capture
    # sin(pi*x/4) maps -2->-1, 0->0, 2->1, with non-uniform spacing
    t = np.sin(np.pi * x / 4.0)
    
    # Compute Chebyshev polynomials T_0 to T_9 in t
    T_prev2 = np.ones_like(t)
    T_prev1 = t
    
    result = params[0] * T_prev2 + params[1] * T_prev1
    
    for k in range(2, 10):
        T_curr = 2.0 * t * T_prev1 - T_prev2
        result = result + params[k] * T_curr
        T_prev2 = T_prev1
        T_prev1 = T_curr
    
    return result
# EVOLVE-BLOCK-END
