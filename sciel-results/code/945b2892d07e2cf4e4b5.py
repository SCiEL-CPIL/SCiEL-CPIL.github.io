"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis model: 5 global smooth functions + 5 local Gaussian RBFs.
    Designed to fit arbitrary smooth functions on [0.1, 6] with 10 parameters.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure finite inputs
    x = np.where(np.isfinite(x), x, 3.0)
    p = np.where(np.isfinite(p), p, 0.0)
    
    # 1. Global Basis Functions
    # Normalized x to help conditioning
    x_norm = (x - 3.05) / 2.95
    
    g0 = np.ones_like(x)
    g1 = x_norm
    g2 = x_norm ** 2
    g3 = np.sin(x)
    g4 = np.cos(x)
    
    # 2. Local Basis Functions (Gaussian RBFs)
    # Centers spread across the domain
    centers = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sigma = 0.8
    
    # Compute Gaussians efficiently
    # x is (N,), centers is (5,)
    # diff is (N, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    l0 = np.exp(-(diff ** 2) / (2 * sigma ** 2))
    
    # Combine into design matrix (N, 10)
    # Columns: g0, g1, g2, g3, g4, l00, l01, l02, l03, l04
    X = np.hstack([
        g0[:, np.newaxis],
        g1[:, np.newaxis],
        g2[:, np.newaxis],
        g3[:, np.newaxis],
        g4[:, np.newaxis],
        l0
    ])
    
    # Prediction
    return X @ p
# EVOLVE-BLOCK-END
