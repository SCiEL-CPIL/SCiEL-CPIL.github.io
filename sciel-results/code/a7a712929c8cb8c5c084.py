"""Symbolic regression with Hybrid Chebyshev + RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: 5 Chebyshev polynomials (global) + 5 Gaussian RBFs (local).
    
    Chebyshev polynomials T_0 to T_4 capture global trends.
    Gaussian RBFs with fixed centers capture local features.
    
    This hybrid approach is robust to both polynomial and non-polynomial targets,
    providing better conditioning and approximation power than pure polynomials.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure finite inputs
    x = np.where(np.isfinite(x), x, 3.0)
    p = np.where(np.isfinite(p), p, 0.0)
    
    # 1. Chebyshev Part (Global)
    # Scale x from [0.1, 6] to t in [-1, 1]
    t = (2.0 * x - 6.1) / 5.9
    t = np.clip(t, -1.0, 1.0)
    
    T0 = np.ones_like(t)
    T1 = t.copy()
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    
    # 2. RBF Part (Local)
    # Centers evenly spaced in [0.1, 6]
    centers = np.array([0.1, 1.675, 3.25, 4.825, 6.0])
    sigma = 1.0
    
    # Compute Gaussian RBFs
    # Shape: (N, 5)
    # x[:, None] - centers[None, :]
    diff = x[:, None] - centers[None, :]
    rbf_values = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Stack basis functions
    # 5 Chebyshev + 5 RBF = 10 basis functions
    X = np.column_stack([T0, T1, T2, T3, T4, rbf_values])
    
    # Prediction
    return X @ p
# EVOLVE-BLOCK-END
