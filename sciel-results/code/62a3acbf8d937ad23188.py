"""Rewritten program for symbolic regression with an RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (RBF) model with 10 parameters.
    f(x, y) = sum_{i=0}^{8} p_i * exp(-gamma * ((x-cx_i)^2 + (y-cy_i)^2)) + p_9
    
    We use 9 Gaussian RBFs centered at a 3x3 grid on [-1, 1]^2 and one bias term.
    The width parameter gamma is fixed to 2.0 (sigma = 0.5).
    This allows the model to capture non-polynomial smooth structures better than
    a cubic polynomial, which is the primary limitation of the previous approach.
    The model remains linear in parameters, ensuring stable optimization with BFGS.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit

    p = params

    # Centers for the 9 RBFs
    centers = [
        (-1.0, -1.0), (0.0, -1.0), (1.0, -1.0),
        (-1.0,  0.0), (0.0,  0.0), (1.0,  0.0),
        (-1.0,  1.0), (0.0,  1.0), (1.0,  1.0)
    ]
    
    gamma = 2.0
    
    # Pre-compute basis functions
    # B0-B8 are RBFs, B9 is bias
    pred = np.zeros_like(x)
    
    for i, (cx, cy) in enumerate(centers):
        dist_sq = (x - cx)**2 + (y - cy)**2
        basis = np.exp(-gamma * dist_sq)
        pred += p[i] * basis
        
    # Bias term
    pred += p[9] * np.ones_like(x)

    # Safety check for finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)

    # Deadline guard
    if time.time() > deadline:
        return pred

    return pred
# EVOLVE-BLOCK-END
