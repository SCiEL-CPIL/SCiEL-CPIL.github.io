"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # RBF basis with fixed centers
    # x is in [0.1, 6]
    # Use 10 RBFs with centers spread across the domain
    # This creates a linear model in params, which BFGS solves robustly
    
    x_min = 0.1
    x_max = 6.0
    n_centers = 10
    
    # Centers uniformly spaced
    centers = np.linspace(x_min, x_max, n_centers)
    
    # Width of the RBFs. 
    # A value of 0.8 allows for some overlap but distinct bumps.
    sigma = 0.8
    
    # Compute RBF values
    # x shape: (N,)
    # centers shape: (10,)
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian RBF
    rbf_values = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # rbf_values shape: (N, 10)
    # params shape: (10,)
    # Prediction shape: (N,)
    prediction = rbf_values @ params
    
    return prediction
# EVOLVE-BLOCK-END
