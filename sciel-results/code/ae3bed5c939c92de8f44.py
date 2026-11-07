"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # x is shape (N,)
    # params is shape (10,)
    
    # Basis 0: Intercept
    # Basis 1: Linear
    # Basis 2-9: Gaussian RBFs
    
    # Define fixed centers for the 8 RBFs
    # Spaced evenly across the domain, leaving a small margin
    centers = np.linspace(-1.8, 1.8, 8)
    
    # Define fixed width (standard deviation)
    sigma = 0.6
    
    # Pre-compute the squared distances for efficiency?
    # x is (N,), centers is (8,)
    # We need (N, 8) matrix of (x_i - c_j)^2
    # x[:, np.newaxis] - centers[np.newaxis, :]
    
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    sq_diff = diff * diff
    
    # Compute RBF values: exp(-sq_diff / (2 * sigma^2))
    rbf_values = np.exp(-sq_diff / (2.0 * sigma * sigma))
    
    # Construct the full basis matrix
    # Shape: (N, 10)
    # Col 0: 1
    # Col 1: x
    # Col 2-9: rbf_values
    
    basis = np.zeros((len(x), 10))
    basis[:, 0] = 1.0
    basis[:, 1] = x
    basis[:, 2:10] = rbf_values
    
    # Return linear combination
    return basis @ params
# EVOLVE-BLOCK-END
