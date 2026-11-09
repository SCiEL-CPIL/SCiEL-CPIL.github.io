"""
Uses a radial basis function (RBF) network with a 3x3 grid of centers.
RBFs provide a flexible, local approximation that can capture complex
structures including periodic and non-polynomial components. The linear
combination of RBF activations is fitted by the evaluator's BFGS optimizer.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure inputs are 1D arrays
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    
    # Define centers for Radial Basis Functions (RBFs)
    # A 3x3 grid covering the domain [-1, 1] x [-1, 1]
    centers = np.array([
        [-1.0, -1.0],
        [ 0.0, -1.0],
        [ 1.0, -1.0],
        [-1.0,  0.0],
        [ 0.0,  0.0],
        [ 1.0,  0.0],
        [-1.0,  1.0],
        [ 0.0,  1.0],
        [ 1.0,  1.0]
    ])
    
    # Width parameter for RBFs (gamma = 1 / (2 * sigma^2))
    # gamma = 5.0 provides a balance between local flexibility and global smoothness
    gamma = 5.0
    
    # Compute distances from each point to each center
    # x: (n,), centers: (9, 2)
    # dist: (n, 9)
    dx = x[:, np.newaxis] - centers[:, 0][np.newaxis, :]
    dy = y[:, np.newaxis] - centers[:, 1][np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    # RBF activations: exp(-gamma * dist_sq)
    rbf_vals = np.exp(-gamma * dist_sq)  # shape: (n, 9)
    
    # Unpack parameters (10 params: 9 for RBFs + 1 bias)
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Linear combination of RBF activations + bias
    # rbf_vals @ [p0..p8] + p9
    prediction = rbf_vals @ params[:9] + params[9]
    
    return prediction
# EVOLVE-BLOCK-END
