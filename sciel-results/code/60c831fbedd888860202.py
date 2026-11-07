"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF-based model with 10 parameters.
    params[0] is a constant offset.
    params[1:10] are weights for 9 RBF centers evenly spaced in [0.1, 6].
    Each RBF is a Gaussian with fixed width.
    """
    # 9 centers evenly spaced in [0.1, 6]
    centers = np.linspace(0.1, 6.0, 9)
    # Fixed width for RBFs - choose a reasonable value
    width = 1.0
    
    # Build RBF matrix: shape (n_points, 9)
    x_expanded = x[:, np.newaxis]  # (n, 1)
    centers_expanded = centers[np.newaxis, :]  # (1, 9)
    
    # Compute squared distances
    dist_sq = (x_expanded - centers_expanded) ** 2  # (n, 9)
    
    # RBF values: exp(-dist_sq / (2 * width^2))
    rbf_values = np.exp(-dist_sq / (2 * width ** 2))  # (n, 9)
    
    # Combine: constant + weighted sum of RBFs
    # params[0] is constant, params[1:10] are RBF weights
    prediction = params[0] + rbf_values @ params[1:10]
    
    return prediction
# EVOLVE-BLOCK-END
