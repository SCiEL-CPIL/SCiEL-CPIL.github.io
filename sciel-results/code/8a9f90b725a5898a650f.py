"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation using a multi-scale Gaussian RBF basis.
    
    This approach uses 10 parameters split into two groups of 5:
    - params[0:5]: Weights for 5 wide Gaussians (sigma=1.0) centered at -2, -1, 0, 1, 2
    - params[5:10]: Weights for 5 narrow Gaussians (sigma=0.4) centered at -1.5, -0.5, 0.5, 1.5, 0
    
    The wide Gaussians capture global trends and broad shapes, while the narrow
    Gaussians capture local sharp features, peaks, or kinks. This multi-scale
    approach provides both global smoothness and local detail, which is more
    flexible than a single-scale RBF basis.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    
    # Define wide Gaussian centers (5 centers)
    wide_centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    wide_sigma = 1.0
    
    # Define narrow Gaussian centers (5 centers)
    narrow_centers = np.array([-1.5, -0.5, 0.5, 1.5, 0.0])
    narrow_sigma = 0.4
    
    # Compute wide Gaussian basis functions
    # Shape: (N, 5)
    diff_wide = x[:, np.newaxis] - wide_centers[np.newaxis, :]
    wide_gaussians = np.exp(-0.5 * (diff_wide / wide_sigma) ** 2)
    
    # Compute narrow Gaussian basis functions
    # Shape: (N, 5)
    diff_narrow = x[:, np.newaxis] - narrow_centers[np.newaxis, :]
    narrow_gaussians = np.exp(-0.5 * (diff_narrow / narrow_sigma) ** 2)
    
    # Combine bases
    # Total shape: (N, 10)
    # First 5 columns are wide Gaussians, next 5 are narrow Gaussians
    basis = np.hstack((wide_gaussians, narrow_gaussians))
    
    # Linear combination
    return basis @ params
# EVOLVE-BLOCK-END
