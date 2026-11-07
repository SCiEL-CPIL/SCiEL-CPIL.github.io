# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs (x, y) drawn from [-1, 1]^2.
    
    Strategy: Use a sum of Gaussian Radial Basis Functions (RBFs).
    This is a structural change from a global polynomial/trig basis to a
    local additive model. 10 parameters are used as weights for 10 fixed
    Gaussian centers. This allows the model to fit complex, non-linear
    surfaces by activating specific regions of the input space.
    """
    p = params
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Define 10 fixed centers covering the domain [-1, 1]^2
    # 3x3 grid from -0.5 to 0.5 plus one corner to extend coverage
    centers = np.array([
        [0.0, 0.0],
        [0.5, 0.0],
        [-0.5, 0.0],
        [0.0, 0.5],
        [0.0, -0.5],
        [0.5, 0.5],
        [-0.5, 0.5],
        [0.5, -0.5],
        [-0.5, -0.5],
        [0.7, 0.7] # Extra center to help with corners/edges
    ])
    
    # Width of the Gaussians. 
    # A value around 0.5-0.7 is typical for this domain size.
    sigma = 0.6
    sigma2 = 2 * sigma * sigma
    
    # Calculate squared distances from each point (x, y) to each center
    # x, y are arrays of shape (N,)
    # centers is (10, 2)
    
    # Efficient calculation:
    # dx = x[:, None] - centers[0, :]  # (N, 10)
    # dy = y[:, None] - centers[1, :]  # (N, 10)
    # dist_sq = dx**2 + dy**2          # (N, 10)
    # gaussians = np.exp(-dist_sq / sigma2) # (N, 10)
    # result = gaussians @ p           # (N,)
    
    dx = x[:, np.newaxis] - centers[:, 0]
    dy = y[:, np.newaxis] - centers[:, 1]
    
    dist_sq = dx * dx + dy * dy
    gaussians = np.exp(-dist_sq / sigma2)
    
    return gaussians @ p
# EVOLVE-BLOCK-END
