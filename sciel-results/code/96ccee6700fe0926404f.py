"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF network with 5 fixed centers, Gaussian kernels.
    
    params layout (length 10):
      params[0:5]  - weights for the 5 RBF kernels
      params[5]    - global offset (bias)
      params[6]    - width scaling factor (applied to inverse of default width)
      params[7:10] - spare (set to 0 for stability, or used for center shifts if needed)
    
    Centers are fixed at equally spaced points in [-3, 3]: -3, -1.5, 0, 1.5, 3
    Default width sigma = 1.0, scaled by 1/params[6] (clamped to avoid division by zero)
    """
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.ravel()
    
    # Fixed centers
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # Extract parameters
    weights = params[0:5]
    offset = params[5]
    width_scale = params[6]
    
    # Clamp width_scale to avoid division by zero or extreme values
    # Use softplus-like clamping: ensure width_scale is in [0.1, 10]
    width_scale_clamped = np.clip(width_scale, 0.1, 10.0)
    
    # Compute Gaussian RBF values
    # diff shape: (n, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    # sigma = 1.0 / width_scale_clamped
    sigma = 1.0 / width_scale_clamped
    rbf_values = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Prediction: sum of weights * rbf_values + offset
    pred = rbf_values @ weights + offset
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
