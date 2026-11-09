"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Fixed-width RBF model with 10 centers evenly spaced across [-3, 3]
    # The 10 params directly represent the RBF weights (linear model in params)
    # This ensures BFGS converges to the global optimum
    
    centers = np.linspace(-3.0, 3.0, 10)
    width = 0.5  # moderate global width
    
    # Compute RBF features: exp(-((x - center)^2) / (2 * width^2))
    # x has shape (n_points,), centers has shape (10,)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # (n_points, 10)
    rbf = np.exp(-diff**2 / (2.0 * width**2))  # (n_points, 10)
    
    # Weighted sum of RBFs
    # params has shape (10,)
    return rbf @ params
# EVOLVE-BLOCK-END
