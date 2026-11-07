"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Universal approximator using a sum of Gaussian RBFs.
    Uses 5 Gaussian basis functions centered at (cx_i, cy_i).
    params[0:5] -> centers (cx1, cx2, cx3, cx4, cx5)
    params[5:10] -> centers (cy1, cy2, cy3, cy4, cy5)
    Weights are fixed to 1 to allow BFGS to optimize only the centers,
    which is a more robust optimization landscape than optimizing both
    weights and centers simultaneously with only 10 parameters.
    """
    # Ensure params is a flat array of length 10
    params = np.asarray(params).flatten()
    
    # If params are invalid or not length 10, return zeros (or ones) to be safe
    if params.size != 10:
        return np.zeros_like(x, dtype=float)
    
    # Extract centers
    cx = params[0:5]
    cy = params[5:10]
    
    # Compute distances from each point (x, y) to each center (cx_i, cy_i)
    # x and y are shape (N,), cx and cy are shape (5,)
    # We want dist_sq[i, j] = (x[i] - cx[j])^2 + (y[i] - cy[j])^2
    dx = x[:, np.newaxis] - cx[np.newaxis, :]  # Shape (N, 5)
    dy = y[:, np.newaxis] - cy[np.newaxis, :]  # Shape (N, 5)
    
    dist_sq = dx**2 + dy**2
    
    # Gaussian RBF: exp(-dist_sq / sigma^2)
    # Fixed sigma = 1.0 (inputs are in [-1, 1], so this is a reasonable scale)
    sigma_sq = 1.0
    
    rbf_values = np.exp(-dist_sq / sigma_sq)
    
    # Sum of RBFs with fixed weight 1
    pred = np.sum(rbf_values, axis=1)
    
    return pred
# EVOLVE-BLOCK-END
