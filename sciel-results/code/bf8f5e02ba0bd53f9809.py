"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x from [-2, 2] to [-1, 1] for Chebyshev polynomials
    t = 0.5 * x
    
    # Compute Chebyshev polynomials T_0(t) ... T_9(t)
    n_params = 10
    basis = np.zeros((len(x), n_params))
    
    # T_0(t) = 1
    basis[:, 0] = 1.0
    
    # T_1(t) = t
    if n_params > 1:
        basis[:, 1] = t
    
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    for k in range(2, n_params):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
    
    # Return the linear combination of Chebyshev basis functions
    return basis @ params
# EVOLVE-BLOCK-END
