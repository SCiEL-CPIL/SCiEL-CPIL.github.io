"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Define RBF centers evenly spaced across the domain [0.1, 6.0]
    x_min = 0.1
    x_max = 6.0
    n_rbf = 10
    
    # Centers
    centers = np.linspace(x_min, x_max, n_rbf)
    
    # Fixed width (sigma) for the Gaussian RBFs
    # Spacing is approx (6.0 - 0.1) / 9 = 0.655
    # Choose sigma to allow good overlap but maintain locality
    sigma = 0.7
    
    # Precompute the squared distances (x_j - c_i)^2
    # x shape: (N, 1)
    # centers shape: (1, M)
    x_reshaped = x.reshape(-1, 1)
    centers_reshaped = centers.reshape(1, -1)
    
    # Compute (x - c)^2
    diff = x_reshaped - centers_reshaped
    diff_sq = diff * diff
    
    # Compute Gaussian RBFs: exp(-diff_sq / (2 * sigma^2))
    # 2 * sigma^2
    denom = 2.0 * sigma * sigma
    rbf_matrix = np.exp(-diff_sq / denom)
    
    # Linear combination: rbf_matrix (N, M) @ params (M,)
    return rbf_matrix @ params
# EVOLVE-BLOCK-END
