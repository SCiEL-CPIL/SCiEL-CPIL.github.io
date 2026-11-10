"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF model with Gaussian kernels on 5 centers: 4 corners and origin.
    params: length-10 array.
      params[0..4]: weights for the 5 RBF centers.
      params[5..9]: bandwidths (sigma) for the 5 RBF centers.
    """
    # Centers: 4 corners and origin in [-1, 1]^2
    cx = np.array([-1.0, 1.0, -1.0, 1.0, 0.0])
    cy = np.array([-1.0, -1.0, 1.0, 1.0, 0.0])
    
    # Extract weights and bandwidths
    weights = params[0:5]
    sigmas = params[5:10]
    
    # Ensure sigmas are positive to avoid division by zero or negative variance
    # Using a small epsilon to prevent numerical issues if optimizer pushes sigma to 0
    sigmas = np.maximum(sigmas, 1e-6)
    
    # Compute squared distances from each input point to each center
    # x, y are arrays of shape (n,)
    # dist_sq[i, j] = (x[i] - cx[j])^2 + (y[i] - cy[j])^2
    dx = x[:, np.newaxis] - cx[np.newaxis, :]  # (n, 5)
    dy = y[:, np.newaxis] - cy[np.newaxis, :]  # (n, 5)
    dist_sq = dx ** 2 + dy ** 2  # (n, 5)
    
    # Gaussian RBF values: exp(-dist_sq / (2 * sigma^2))
    # sigmas is shape (5,), need to broadcast to (1, 5)
    sigma_sq = sigmas ** 2
    rbf_values = np.exp(-dist_sq / (2.0 * sigma_sq[np.newaxis, :]))  # (n, 5)
    
    # Prediction: sum of weighted RBFs
    pred = rbf_values @ weights
    
    return pred
# EVOLVE-BLOCK-END
