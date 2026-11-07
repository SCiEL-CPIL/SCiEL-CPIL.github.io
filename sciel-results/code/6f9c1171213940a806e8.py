"""Rewritten program: Balanced Chebyshev + RBF basis."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE_S = 150.0
_MARGIN_S = 5.0
_START_TIME = time.time()

def _compute_basis(x):
    """
    Computes a balanced basis of 10 functions.
    5 Chebyshev polynomials (T0-T4) for global trend.
    5 Gaussian RBFs for local adjustments.
    """
    xc = np.clip(x, -2.0, 2.0)
    
    # 1. Chebyshev Polynomials T0 to T4
    t = xc * 0.5  # Scale to [-1, 1]
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * t - 1.0
    T3 = 4.0 * t * t * t - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    
    # 2. Gaussian RBFs
    # Centers spread across [-1.5, 1.5]
    centers = np.array([-1.5, -0.75, 0.0, 0.75, 1.5])
    sigma = 0.45
    
    # Compute RBFs: exp(-((x - c)^2) / (2 * sigma^2))
    # Shape: (N, 5)
    rbf_terms = np.exp(-((xc[:, np.newaxis] - centers[np.newaxis, :])**2) / (2.0 * sigma**2))
    
    # Combine into a single basis matrix (N, 10)
    basis = np.column_stack((
        T0, T1, T2, T3, T4,
        rbf_terms
    ))
    
    return basis

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE_S - _MARGIN_S:
        # Fallback: Linear fit using first two params
        xc = np.clip(x, -2.0, 2.0)
        return params[1] * xc + params[0]

    basis = _compute_basis(x)
    
    # Linear combination
    pred = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        xc = np.clip(x, -2.0, 2.0)
        return params[1] * xc + params[0]

    return pred
# EVOLVE-BLOCK-END
