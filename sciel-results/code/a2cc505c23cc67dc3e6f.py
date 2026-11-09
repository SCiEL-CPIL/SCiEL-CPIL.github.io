"""Symbolic regression with composite nonlinear parametric model."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _chebyshev_basis(x: np.ndarray, degree: int) -> np.ndarray:
    """
    Compute Chebyshev polynomial basis of given degree.
    Returns array of shape (len(x), degree + 1) where column k is T_k(x_normalized).
    x_normalized is mapped from [-3, 3] to [-1, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    # Normalize x from [-3, 3] to [-1, 1]
    t = x / 3.0
    
    n = len(x)
    basis = np.zeros((n, degree + 1), dtype=np.float64)
    basis[:, 0] = 1.0
    if degree >= 1:
        basis[:, 1] = t
        for k in range(2, degree + 1):
            basis[:, k] = 2.0 * t * basis[:, k - 1] - basis[:, k - 2]
    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear Chebyshev polynomial basis of degree 9 (10 parameters).
    
    y = sum_{k=0}^{9} params[k] * T_k(x/3)
    
    This is a linear model in params, so BFGS will converge quickly.
    Chebyshev polynomials provide a well-conditioned basis for approximation
    on a bounded interval, avoiding the ill-conditioning of monomial powers.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Compute Chebyshev basis of degree 9
    basis = _chebyshev_basis(x, degree=9)
    
    # Linear combination
    y = basis @ params
    
    # Ensure finite output
    y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return y
# EVOLVE-BLOCK-END
