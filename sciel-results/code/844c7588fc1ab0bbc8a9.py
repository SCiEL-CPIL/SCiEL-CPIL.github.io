"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

_START_TIME = time.time()
_DEADLINE = 145.0

def _chebyshev_basis(x, degree):
    """
    Compute Chebyshev polynomials of the first kind up to 'degree' for scaled x.
    x is assumed to be in [-3, 3]. We scale to [-1, 1] for better conditioning.
    """
    x_scaled = x / 3.0  # Map [-3, 3] to [-1, 1]
    T = np.zeros((x.shape[0], degree + 1), dtype=np.float64)
    T[:, 0] = 1.0
    if degree >= 1:
        T[:, 1] = x_scaled
        for k in range(2, degree + 1):
            T[:, k] = 2 * x_scaled * T[:, k-1] - T[:, k-2]
    return T

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Sum of Chebyshev polynomials T_0(x) to T_9(x).
    params[0:10] are the coefficients for these polynomials.
    
    This is a linear model in params, which is ideal for BFGS if the basis is well-conditioned.
    Chebyshev polynomials on [-1, 1] are orthogonal and bounded, leading to a well-conditioned 
    design matrix compared to monomials.
    """
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x, dtype=np.float64)
    
    x = np.asarray(x, dtype=np.float64)
    
    # We have 10 params, so we use degree 9 (T_0 to T_9)
    degree = 9
    basis = _chebyshev_basis(x, degree)
    
    # Linear combination: y = sum(params[i] * T_i(x))
    # basis is (N, 10), params is (10,)
    pred = basis @ params
    
    # Ensure finite values
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(x, dtype=np.float64)
        
    return pred
# EVOLVE-BLOCK-END
