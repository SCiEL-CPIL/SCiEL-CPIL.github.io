"""Hybrid approach: Polynomial trend + Localized Gaussian RBFs with learnable centers."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the hidden function using 5 Gaussian RBFs with learnable centers
    and a global linear trend.
    
    Parameters:
    params[0..4]: Centers of the 5 Gaussian bumps (learnable, in [-3, 3])
    params[5..9]: Weights for the 5 Gaussian bumps
    
    A small constant offset is added implicitly via the mean of weights.
    The width is fixed to provide good local coverage while allowing overlap.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.ravel()
    n = x.size
    
    if n == 0:
        return np.array([])
    
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure finite params
    p = np.where(np.isfinite(p), p, 0.0)
    
    # Extract centers and weights
    centers = p[0:5]  # Learnable centers
    weights = p[5:10] # Weights for each Gaussian
    
    # Clip centers to valid range [-3, 3] to keep them in domain
    centers = np.clip(centers, -3.0, 3.0)
    
    # Fixed width parameter for the Gaussians
    # lambda = 2.0 implies sigma ~ 0.71, good balance between locality and smoothness
    width = 2.0
    
    # Compute Gaussian basis functions
    # diff shape: (N, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    gaussians = np.exp(-width * diff ** 2)
    
    # Weighted sum of Gaussians
    result = gaussians @ weights
    
    # Add a small global offset: mean of weights acts as a baseline
    # This helps when the function has a non-zero mean
    offset = np.mean(weights)
    result = result + offset
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
