"""Symbolic regression with a hybrid global and local basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis model: 4 global smooth functions + 6 local Gaussian RBFs.
    
    Global basis:
    1. Constant (1)
    2. Linear (x)
    3. Quadratic (x^2)
    4. Sinusoidal (sin(x)) - captures periodic behavior
    
    Local basis:
    6 Gaussian RBFs centered at equally spaced points in [0.1, 6].
    Fixed width sigma = 1.0 for stable conditioning.
    
    This hybrid approach allows BFGS to fit both global trends and local
    variations, improving over pure polynomial bases for complex targets.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure finite inputs
    x = np.where(np.isfinite(x), x, 3.0)
    p = np.where(np.isfinite(p), p, 0.0)
    
    # 1. Global Basis Functions
    # T0: Constant
    g0 = np.ones_like(x)
    # T1: Linear
    g1 = x
    # T2: Quadratic
    g2 = x**2
    # T3: Sinusoidal (scaled to avoid large amplitudes if x is large, but x is small)
    g3 = np.sin(x)
    
    # 2. Local Basis Functions (Gaussian RBFs)
    # Centers distributed across [0.1, 6]
    # We have 6 RBFs. Let's place them at:
    # 0.5, 1.5, 2.5, 3.5, 4.5, 5.5
    centers = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    sigma = 1.0  # Fixed width for stability
    
    rbf_terms = []
    for c in centers:
        # exp(-0.5 * ((x - c) / sigma)^2)
        diff = x - c
        rbf = np.exp(-0.5 * (diff / sigma) ** 2)
        rbf_terms.append(rbf)
    
    # Stack all basis functions into a design matrix (N, 10)
    # Order: g0, g1, g2, g3, rbf1, ..., rbf6
    X = np.column_stack([g0, g1, g2, g3] + rbf_terms)
    
    # Prediction
    pred = X @ p
    
    # Safety check: if prediction is not finite, return zeros
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(x)
        
    return pred
# EVOLVE-BLOCK-END
