"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Multi-scale Radial Basis Function (RBF) network with 10 parameters.
    
    We split the 10 parameters into two groups of 5:
    - params[0:5] -> weights for 5 wide Gaussian kernels (gamma=0.5)
    - params[5:10] -> weights for 5 narrow Gaussian kernels (gamma=5.0)
    
    This allows the model to capture both global trends and local variations.
    
    Wide centers (5):
    (-1, -1), (1, -1), (-1, 1), (1, 1), (0, 0)
    
    Narrow centers (5):
    (-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)
    """
    
    # Define wide centers (5 points)
    wide_centers = np.array([
        [-1.0, -1.0],
        [ 1.0, -1.0],
        [-1.0,  1.0],
        [ 1.0,  1.0],
        [ 0.0,  0.0]
    ])
    
    # Define narrow centers (5 points)
    narrow_centers = np.array([
        [-1.0,  0.0],
        [ 1.0,  0.0],
        [ 0.0, -1.0],
        [ 0.0,  1.0],
        [ 0.0,  0.0]
    ])
    
    # x and y are arrays of shape (N,)
    
    # Compute distances to wide centers
    dx_w = x[:, np.newaxis] - wide_centers[:, 0][np.newaxis, :]
    dy_w = y[:, np.newaxis] - wide_centers[:, 1][np.newaxis, :]
    dist_sq_w = dx_w**2 + dy_w**2
    
    # Compute distances to narrow centers
    dx_n = x[:, np.newaxis] - narrow_centers[:, 0][np.newaxis, :]
    dy_n = y[:, np.newaxis] - narrow_centers[:, 1][np.newaxis, :]
    dist_sq_n = dx_n**2 + dy_n**2
    
    # Gamma values
    gamma_wide = 0.5
    gamma_narrow = 5.0
    
    # RBF values
    rbf_wide = np.exp(-gamma_wide * dist_sq_w)   # Shape (N, 5)
    rbf_narrow = np.exp(-gamma_narrow * dist_sq_n) # Shape (N, 5)
    
    # params shape (10,)
    # Split params
    w_wide = params[0:5]
    w_narrow = params[5:10]
    
    # Prediction
    pred = rbf_wide @ w_wide + rbf_narrow @ w_narrow
    
    return pred
# EVOLVE-BLOCK-END
