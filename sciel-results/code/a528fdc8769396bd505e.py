"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale input to [-1, 1] to improve numerical conditioning
    # x is in [-3, 3], so x/3 is in [-1, 1]
    u = x / 3.0
    
    # Precompute Chebyshev polynomials of the first kind T_0(u) to T_9(u)
    # T_0(u) = 1
    # T_1(u) = u
    # T_k(u) = 2*u*T_{k-1}(u) - T_{k-2}(u)
    
    # T is a matrix of shape (N, 10)
    T = np.empty((len(x), 10), dtype=float)
    T[:, 0] = 1.0
    T[:, 1] = u
    
    for k in range(2, 10):
        T[:, k] = 2.0 * u * T[:, k - 1] - T[:, k - 2]
        
    # The model is a linear combination of these basis functions
    # y = sum_{k=0}^{9} params[k] * T_k(x)
    return T @ params
# EVOLVE-BLOCK-END
