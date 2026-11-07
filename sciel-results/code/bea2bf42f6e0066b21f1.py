"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # 10 fixed RBF centers: 3x3 grid plus an extra point
    # Grid points in [-1, 1] x [-1, 1]
    centers = np.array([
        [-1.0, -1.0],
        [ 0.0, -1.0],
        [ 1.0, -1.0],
        [-1.0,  0.0],
        [ 0.0,  0.0],
        [ 1.0,  0.0],
        [-1.0,  1.0],
        [ 0.0,  1.0],
        [ 1.0,  1.0],
        [ 0.5,  0.5],  # extra point to break symmetry and improve conditioning
    ])
    
    # Compute distances from each point to each center
    # x and y are arrays of shape (N,)
    # centers is shape (10, 2)
    # We want dist[i, j] = ||(x[i], y[i]) - centers[j]||
    
    x_col = x[:, np.newaxis]  # (N, 1)
    y_col = y[:, np.newaxis]  # (N, 1)
    
    cx = centers[:, 0]  # (10,)
    cy = centers[:, 1]  # (10,)
    
    # Compute squared distances
    dx = x_col - cx[np.newaxis, :]  # (N, 10)
    dy = y_col - cy[np.newaxis, :]  # (N, 10)
    dist_sq = dx**2 + dy**2  # (N, 10)
    
    # Gaussian RBF: exp(-beta * dist_sq)
    # Use a fixed beta. Beta=2.0 provides a narrower kernel (width ~0.7),
    # which reduces collinearity between adjacent grid centers and may
    # improve the conditioning of the optimization problem for BFGS.
    beta = 2.0
    
    rbf_values = np.exp(-beta * dist_sq)  # (N, 10)
    
    # Combine with params (weights)
    # params shape (10,)
    # prediction = rbf_values @ params
    pred = rbf_values @ params
    
    return pred
# EVOLVE-BLOCK-END
