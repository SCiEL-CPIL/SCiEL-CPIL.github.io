"""Program using RBF kernel basis for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form symbolic regression candidate using Gaussian RBF kernels.
    10 RBF kernels with fixed, evenly spaced centers over the input domain [-3, 3].
    The 10 parameters act as adaptive weights for local interpolation.
    """
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Define 10 evenly spaced centers over [-3, 3]
    # Use np.linspace with 10 points
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Choose a bandwidth (sigma) for the RBF kernels
    # A moderate bandwidth allows local flexibility while maintaining smoothness
    sigma = 1.0
    
    # Compute the RBF basis matrix
    # For each point in x, compute the RBF value for each center
    # Shape: (n_points, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination of RBF kernels
    y = rbf @ params
    
    # Ensure finite output
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
# EVOLVE-BLOCK-END
