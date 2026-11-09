"""
Uses a radial basis function (RBF) network with 10 fixed centers and widths.
RBFs provide a flexible, local approximation that can capture complex structures
including periodic and non-polynomial components. The linear combination of RBF
activations is fitted by the evaluator's BFGS optimizer.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure inputs are 1D arrays
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    
    # 10 fixed RBF centers spread across [-1, 1] x [-1, 1]
    # Using a grid-like pattern with some offsets for coverage
    centers = np.array([
        [-0.8, -0.8],
        [ 0.0, -0.8],
        [ 0.8, -0.8],
        [-0.8,  0.0],
        [ 0.0,  0.0],
        [ 0.8,  0.0],
        [-0.8,  0.8],
        [ 0.0,  0.8],
        [ 0.8,  0.8],
        [ 0.0,  0.0]  # duplicate center with different width
    ])
    
    # 10 fixed widths (inverse of Gaussian width parameter)
    # Smaller width = more localized, larger width = more global
    widths = np.array([
        1.5, 1.5, 1.5,
        1.5, 2.0, 1.5,
        1.5, 1.5, 1.5,
        3.0   # wider center for global trend
    ])
    
    # Compute squared distances from each point to each center
    # x[:, None] has shape (n_points, 1), centers[None, :, :] has shape (1, n_centers, 2)
    dx = x[:, None] - centers[None, :, 0]
    dy = y[:, None] - centers[None, :, 1]
    dist_sq = dx**2 + dy**2
    
    # Compute Gaussian RBF activations: exp(-width^2 * dist_sq)
    # widths[None, :] has shape (1, n_centers)
    activations = np.exp(-(widths[None, :]**2) * dist_sq)
    
    # Linear combination of activations with learned weights
    # activations has shape (n_points, 10), params has shape (10,)
    return activations @ params
# EVOLVE-BLOCK-END
