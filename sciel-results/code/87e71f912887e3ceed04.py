"""Symbolic regression: recover hidden function via basis expansion fit by BFGS."""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial basis function (RBF) model with 5 fixed centers.
    
    Parameters (10 total):
      - params[0:5]: weights for each RBF center
      - params[5:10]: width modulators for each Gaussian (positive widths)
    
    Centers are evenly spaced across [-3, 3]: -3, -1.5, 0, 1.5, 3
    
    Each basis function is: w_i * exp(-((x - c_i)^2) / (2 * sigma_i^2))
    where sigma_i = 0.5 + 0.5 * (1 + tanh(params[5+i]))  (range [0.5, 1.5])
    
    This smooth, non-polynomial representation can capture localized features
    and is well-conditioned for BFGS optimization.
    """
    weights = params[0:5]
    width_mods = params[5:10]
    
    # Fixed centers evenly spaced across [-3, 3]
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # Compute Gaussian widths: sigma_i = 0.5 + 0.5 * (1 + tanh(width_mods[i]))
    # This ensures sigma is in [0.5, 1.5], keeping it positive and bounded
    sigmas = 0.5 + 0.5 * (1.0 + np.tanh(width_mods))
    
    # Build the RBF expansion
    # x shape: (N,), centers shape: (5,)
    # We want a (N, 5) matrix of RBF values
    x_col = x[:, np.newaxis]  # (N, 1)
    centers_row = centers[np.newaxis, :]  # (1, 5)
    sigmas_row = sigmas[np.newaxis, :]  # (1, 5)
    
    # Compute squared distances: (N, 5)
    diff = x_col - centers_row  # (N, 5)
    sq_dist = diff * diff  # (N, 5)
    
    # Compute Gaussian values: exp(-sq_dist / (2 * sigma^2))
    # (N, 5)
    gaussian_vals = np.exp(-sq_dist / (2.0 * sigmas_row * sigmas_row))
    
    # Combine with weights: (N, 5) @ (5,) -> (N,)
    result = gaussian_vals @ weights
    
    return result
