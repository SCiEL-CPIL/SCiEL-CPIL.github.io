"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers a function using a multi-scale RBF basis with fixed centers.
    The 10 parameters are the weights for 10 Gaussian radial basis functions
    centered at fixed locations in [-1, 1]^2. This allows the model to
    approximate smooth nonlinear functions with local structure.
    
    Centers are chosen to cover the domain:
    - 1 constant term (infinite width)
    - 9 Gaussian centers spread across the square
    
    The widths are fixed to provide multi-scale resolution.
    """
    # Ensure x and y are arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Define fixed centers: 1 constant + 9 Gaussians
    # Centers arranged in a 3x3 grid
    centers = [
        (0.0, 0.0),
        (-0.7, -0.7), (0.0, -0.7), (0.7, -0.7),
        (-0.7, 0.0),  (0.0, 0.0),  (0.7, 0.0),
        (-0.7, 0.7),  (0.0, 0.7),  (0.7, 0.7),
    ]
    
    # Widths: mix of global and local scales
    # First basis function is constant (width = inf, effectively just a bias)
    # Others use different widths for multi-scale approximation
    widths = [
        1e10,  # constant term
        0.4, 0.3, 0.4,
        0.3, 0.25, 0.3,
        0.4, 0.3, 0.4,
    ]
    
    # Compute basis values
    basis = np.empty((x.shape[0], 10), dtype=float)
    
    for i in range(10):
        cx, cy = centers[i]
        w = widths[i]
        if w > 1e9:
            basis[:, i] = 1.0
        else:
            dist_sq = (x - cx)**2 + (y - cy)**2
            basis[:, i] = np.exp(-dist_sq / (2.0 * w * w))
    
    # Linear combination
    return np.einsum('ij,j->i', basis, params)
# EVOLVE-BLOCK-END
