"""Symbolic regression with 1 input; Chebyshev polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _chebyshev_basis(x: np.ndarray, n_terms: int = 10) -> np.ndarray:
    """
    Compute Chebyshev polynomial basis T_k(t) for k=0..n_terms-1,
    where t is x mapped from [0.1, 6] to [-1, 1].
    Uses the recurrence T_0=1, T_1=t, T_k=2*t*T_{k-1}-T_{k-2}.
    Returns shape (n_points, n_terms).
    """
    x = np.asarray(x, dtype=float)
    # Map x from [0.1, 6] to [-1, 1]
    t = (2.0 * x - (6.0 + 0.1)) / (6.0 - 0.1)
    t = np.clip(t, -1.0, 1.0)  # safety clip
    
    n = len(t)
    basis = np.zeros((n, n_terms))
    if n_terms > 0:
        basis[:, 0] = 1.0
    if n_terms > 1:
        basis[:, 1] = t
    for k in range(2, n_terms):
        basis[:, k] = 2.0 * t * basis[:, k - 1] - basis[:, k - 2]
    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial model of degree 9 (10 coefficients).
    
    y = sum_{k=0}^{9} p_k * T_k(t), where t = (2x - 6.1) / 5.9
    
    This is a linear model in params, so BFGS converges in one step.
    Chebyshev basis is orthogonal on [-1,1], giving excellent conditioning.
    Can represent any smooth function on [0.1, 6] to high accuracy.
    """
    x = np.asarray(x, dtype=float)
    basis = _chebyshev_basis(x, n_terms=10)
    return basis @ params
# EVOLVE-BLOCK-END
