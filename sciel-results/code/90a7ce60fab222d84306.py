"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial basis with 10 parameters.
    Maps x from [0.1, 6] to t in [-1, 1] and uses T_0(t) to T_9(t).
    This provides a well-conditioned global basis for smooth functions.
    """
    # Map x from [0.1, 6] to t in [-1, 1]
    # t = (2x - (min + max)) / (max - min)
    x_min, x_max = 0.1, 6.0
    t = (2.0 * x - (x_min + x_max)) / (x_max - x_min)
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_{k+1}(t) = 2t T_k(t) - T_{k-1}(t)
    
    n_basis = 10
    basis = np.zeros((len(x), n_basis))
    
    # T_0
    basis[:, 0] = 1.0
    if n_basis > 1:
        # T_1
        basis[:, 1] = t
        for k in range(2, n_basis):
            basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
            
    # Linear combination
    result = basis @ params
    
    return result
# EVOLVE-BLOCK-END
