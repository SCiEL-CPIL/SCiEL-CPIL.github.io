"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Use 10 Radial Basis Functions (Gaussians) centered at evenly spaced points
    across [-3, 3]. This provides localized flexibility to approximate
    non-polynomial, non-periodic smooth targets that global polynomial bases
    struggle with.
    
    Centers: -3, -2.4, -1.8, -1.2, -0.6, 0, 0.6, 1.2, 1.8, 2.4
    Width: sigma = 1.0 (fixed, chosen for good coverage with 10 centers)
    """
    x = np.asarray(x, dtype=float)
    
    # Define 10 centers evenly spaced in [-3, 3]
    centers = np.linspace(-3.0, 2.4, 10)
    
    # Fixed width for Gaussian RBFs
    sigma = 1.0
    
    # Compute the RBF basis matrix: shape (n_points, 10)
    # basis[i, j] = exp(-(x[i] - centers[j])^2 / (2 * sigma^2))
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    basis = np.exp(-diff ** 2 / (2.0 * sigma ** 2))
    
    return basis @ params
# EVOLVE-BLOCK-END
