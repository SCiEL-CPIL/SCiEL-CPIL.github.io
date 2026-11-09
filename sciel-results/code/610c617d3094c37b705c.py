"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a Radial Basis Function (RBF) model.
    The model is:
    p0 + sum_{i=0}^{2} sum_{j=0}^{2} p_{1+3i+j} * exp(-beta * ((x - cx_i)^2 + (y - cy_j)^2))
    
    Centers are fixed at {-1, 0, 1} for both x and y.
    Beta is fixed at 2.0 to control the width of the RBFs.
    This provides a flexible, smooth approximation that is linear in parameters,
    ensuring convex optimization.
    """
    # Fixed parameters for the RBF basis
    beta = 2.0
    centers = [-1.0, 0.0, 1.0]
    
    pred = np.full_like(x, params[0], dtype=float)
    
    idx = 1
    for cx in centers:
        for cy in centers:
            # Calculate squared distance from each point (x, y) to center (cx, cy)
            dx = x - cx
            dy = y - cy
            dist_sq = dx * dx + dy * dy
            
            # Compute RBF value
            rbf_val = np.exp(-beta * dist_sq)
            
            # Add weighted RBF to prediction
            pred += params[idx] * rbf_val
            idx += 1
            
    return pred
# EVOLVE-BLOCK-END
