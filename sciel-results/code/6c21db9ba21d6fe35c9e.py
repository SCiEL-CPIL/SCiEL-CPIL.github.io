"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Use Radial Basis Functions (Gaussians) with fixed centers
    # This is a local basis, good for non-polynomial smooth functions
    centers = np.linspace(-2.0, 2.0, 10)
    sigma = 0.5
    
    # Compute the RBF values
    # x is (N,), centers is (10,)
    # We want (N, 10) matrix of distances
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernel
    rbf = np.exp(-diff**2 / (2.0 * sigma**2))
    
    # Linear combination of RBFs
    return rbf @ params
# EVOLVE-BLOCK-END
