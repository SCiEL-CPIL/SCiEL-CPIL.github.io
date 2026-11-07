"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Fixed centers for Gaussian RBFs spread across [-2, 2]
_RBF_CENTERS = np.linspace(-2.0, 2.0, 10)
_RBF_WIDTH = 0.8  # Width parameter for the Gaussians

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF model: sum of 10 weighted Gaussian basis functions.
    
    Each basis function is a Gaussian centered at a fixed point in [-2, 2].
    This provides a flexible, local approximation that can represent
    a wide variety of smooth functions.
    
    params: array of 10 coefficients (weights for each RBF).
    x: array of input points.
    """
    x = np.asarray(x, dtype=float)
    
    # Compute RBF values: for each center, exp(-((x - center)^2) / (2 * width^2))
    # Shape: (n_points, 10)
    diff = x[:, np.newaxis] - _RBF_CENTERS[np.newaxis, :]
    rbf_values = np.exp(-diff**2 / (2.0 * _RBF_WIDTH**2))
    
    # Linear combination: (n_points, 10) @ (10,) -> (n_points,)
    result = rbf_values @ params
    
    return result
# EVOLVE-BLOCK-END
