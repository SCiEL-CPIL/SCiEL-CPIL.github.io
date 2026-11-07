"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters basis of 10 Chebyshev polynomials of the first kind.
    The input x is mapped from [-3, 3] to [-1, 1] for Chebyshev evaluation.
    params[0..9] are coefficients for T_0(x'), T_1(x'), ..., T_9(x').
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map from [-3, 3] to [-1, 1]
    # x' = (x - 0) / 3 = x / 3
    # Actually, to map [-3,3] to [-1,1]: x' = (x - mid) / half_range
    # mid = 0, half_range = 3
    # So x' = x / 3
    x_prime = x / 3.0
    
    # Compute Chebyshev polynomials T_0 through T_9
    # T_0(x) = 1
    # T_1(x) = x
    # T_{k+1}(x) = 2*x*T_k(x) - T_{k-1}(x)
    
    T = np.empty((10, x.shape[0]), dtype=np.float64)
    T[0] = 1.0
    if x.shape[0] > 0:
        T[1] = x_prime
        for k in range(2, 10):
            T[k] = 2.0 * x_prime * T[k-1] - T[k-2]
    
    # Compute prediction: sum of params[k] * T[k]
    # params is shape (10,), T is shape (10, n)
    # result = params @ T
    pred = params @ T
    
    return pred
# EVOLVE-BLOCK-END
