"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    n_points = len(x)
    if n_points == 0:
        return np.array([])
    
    # Fixed-width RBF model with 10 centers evenly spaced across [-3, 3]
    # Centers: -3, -2.4, -1.8, -1.2, -0.6, 0, 0.6, 1.2, 1.8, 2.4
    # Wait, 10 centers evenly spaced from -3 to 3:
    # spacing = 6 / 9 = 0.666...
    # centers = np.linspace(-3, 3, 10)
    
    centers = np.linspace(-3.0, 3.0, 10)
    width = 0.5
    
    # Compute RBF values: phi_j(x_i) = exp(-((x_i - c_j)^2) / (2 * width^2))
    # x has shape (n_points,), centers has shape (10,)
    # We need a matrix of shape (n_points, 10) or (10, n_points)
    
    # diff[i, j] = x[i] - centers[j]
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_matrix = np.exp(-0.5 * (diff ** 2) / (width ** 2))
    
    # rbf_matrix has shape (n_points, 10)
    # params has shape (10,)
    # prediction[i] = sum_j rbf_matrix[i, j] * params[j]
    return rbf_matrix @ params
# EVOLVE-BLOCK-END
