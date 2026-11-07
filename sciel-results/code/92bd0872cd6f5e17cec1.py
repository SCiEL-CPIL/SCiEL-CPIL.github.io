"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Uses a Gaussian Radial Basis Function (RBF) network with fixed centers and widths.
    The model is linear in the parameters (weights), which ensures that the BFGS
    optimizer converges reliably to the global minimum of the linear least squares
    problem, avoiding local minima issues associated with non-linear basis functions
    or high-degree polynomials.
    
    We use 10 RBFs centered at various points in the [-1, 1]^2 domain.
    The width (gamma) is fixed to 3.0 to balance locality and global coverage.
    """
    # Fixed centers for the RBFs
    centers = np.array([
        [-1.0, -1.0],
        [ 1.0, -1.0],
        [-1.0,  1.0],
        [ 1.0,  1.0],
        [ 0.0,  0.0],
        [-0.5,  0.0],
        [ 0.5,  0.0],
        [ 0.0, -0.5],
        [ 0.0,  0.5],
        [-0.5, -0.5]
    ])
    
    # Fixed width parameter
    gamma = 3.0
    
    # Compute distances from each point (x, y) to each center
    # x and y are arrays of shape (N,)
    # centers is (10, 2)
    # We want dist_sq[i, j] = (x[i] - c[j, 0])^2 + (y[i] - c[j, 1])^2
    
    x_col = x[:, np.newaxis]  # (N, 1)
    y_col = y[:, np.newaxis]  # (N, 1)
    
    cx = centers[:, 0][np.newaxis, :]  # (1, 10)
    cy = centers[:, 1][np.newaxis, :]  # (1, 10)
    
    dist_sq = (x_col - cx)**2 + (y_col - cy)**2  # (N, 10)
    
    # Compute RBF values
    rbf_values = np.exp(-gamma * dist_sq)  # (N, 10)
    
    # Linear combination with params
    # params is (10,)
    prediction = rbf_values @ params
    
    return prediction
# EVOLVE-BLOCK-END
