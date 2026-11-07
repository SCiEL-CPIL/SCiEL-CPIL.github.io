"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (RBF) model using Gaussian kernels.
    params: array of length 10, representing weights for each kernel.
    """
    # Define fixed centers and widths for the RBFs
    # Centers evenly spaced in [-2, 2]
    centers = np.linspace(-2.0, 2.0, 10)
    # Fixed width (sigma) for the Gaussian kernels
    sigma = 0.6
    
    # Compute the RBF matrix
    # x is shape (N,), centers is shape (10,)
    # We want a matrix of shape (N, 10) where M[i, j] = exp(-0.5 * ((x[i] - centers[j]) / sigma)^2)
    
    # Reshape x for broadcasting
    x_col = x[:, np.newaxis]
    
    # Compute squared distances
    # diff shape: (N, 10)
    diff = x_col - centers[np.newaxis, :]
    
    # Compute Gaussian values
    # Using the standard Gaussian kernel form: exp(-0.5 * (d/sigma)^2)
    rbf_matrix = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination with weights (params)
    # rbf_matrix shape (N, 10), params shape (10,)
    return rbf_matrix @ params
# EVOLVE-BLOCK-END
