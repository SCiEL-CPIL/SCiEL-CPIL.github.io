"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Fixed centers for Gaussian basis functions
    # 10 parameters correspond to 10 Gaussian weights
    # Centers evenly spaced in [-2, 2]
    centers = np.linspace(-2.0, 2.0, 10)
    # Width of the Gaussians. 
    # A width of 1.0 allows for smooth interpolation across the domain.
    # Smaller width (e.g., 0.5) allows for sharper features but might be less stable.
    # Let's try 1.0 first.
    sigma = 1.0
    
    # Compute the basis matrix: B[i, j] = exp(-(x[i] - centers[j])^2 / (2 * sigma^2))
    # x is shape (N,), centers is shape (10,)
    # We want result shape (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    basis = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Prediction is params @ basis.T
    # params is shape (10,), basis.T is shape (10, N)
    # Result is shape (N,)
    return params @ basis.T
# EVOLVE-BLOCK-END
