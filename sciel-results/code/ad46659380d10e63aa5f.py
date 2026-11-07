"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Sparse linear combination of 10 Radial Basis Functions (RBFs).
    
    Centers and widths are fixed by a deterministic Halton-like grid over [-1, 1]^2.
    The optimizer fits only the 10 amplitudes (params).
    
    x: array of shape (n, 2) or two arrays? 
       Looking at the seed: params[0] * x + params[1] * y + params[2]
       This implies x and y are separate 1D arrays of length n.
       Let's verify: "equation(x, y, params)".
       The seed uses x and y directly as scalars/vectors.
       So x is shape (n,), y is shape (n,).
    """
    # Flatten just in case, though they should be 1D
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    params = np.asarray(params).ravel()
    
    n = len(x)
    
    # Define 10 fixed centers in [-1, 1] x [-1, 1]
    # Using a deterministic grid or Halton sequence.
    # Let's use a 3x3 grid plus 1 center to make 10 points, or just a 4x2 grid?
    # A 3x3 grid has 9 points. Let's add the center (0,0) to make 10.
    # Or use a Halton sequence for better space-filling.
    
    # Halton sequence in 2D for first 10 points (skipping 0,0 usually, or including?)
    # Let's just hardcode 10 well-distributed points.
    # Grid: -1, 0, 1
    # Points:
    # (-1, -1), (-1, 0), (-1, 1)
    # ( 0, -1), ( 0, 0), ( 0, 1)
    # ( 1, -1), ( 1, 0), ( 1, 1)
    # ( 0, 0) is already there. We need 10.
    # Let's add (0.5, 0.5) or something?
    # Actually, a 4x3 grid? No, let's stick to a simple robust set.
    
    # Let's use a 3x3 grid (9 points) + (0,0) is duplicate.
    # Let's use 10 points from a Halton sequence or just a dense grid.
    # 5x2 grid?
    # xs = [-1, -0.5, 0, 0.5, 1]
    # ys = [-0.75, 0.75] -> 10 points.
    
    centers = np.array([
        [-1.0, -0.75],
        [-0.5, -0.75],
        [ 0.0, -0.75],
        [ 0.5, -0.75],
        [ 1.0, -0.75],
        [-1.0,  0.75],
        [-0.5,  0.75],
        [ 0.0,  0.75],
        [ 0.5,  0.75],
        [ 1.0,  0.75]
    ])
    
    # Fixed widths.
    # Width should be large enough to cover the domain but small enough to be localized.
    # Domain is [-1, 1]. Diameter 2.
    # Let's use a width sigma such that the RBF has significant influence over ~0.5 units.
    # exp(-((x-c)^2 + (y-c)^2) / (2*sigma^2))
    # If sigma = 0.5, then at distance 1, exp(-1/0.5) = exp(-2) ~ 0.13.
    # If sigma = 0.7, at distance 1, exp(-1/0.98) ~ 0.36.
    # Let's try sigma = 0.6 for all centers.
    sigma = 0.6
    widths = np.full(10, sigma)
    
    # Compute distances from each point (x_i, y_i) to each center (c_jx, c_jy)
    # x: (n,), y: (n,)
    # centers: (10, 2)
    
    # dist_sq[i, j] = (x[i] - centers[j,0])^2 + (y[i] - centers[j,1])^2
    # Using broadcasting:
    # x_col = x[:, np.newaxis]  # (n, 1)
    # y_col = y[:, np.newaxis]  # (n, 1)
    # cx = centers[:, 0]  # (10,)
    # cy = centers[:, 1]  # (10,)
    
    # dist_sq = (x_col - cx[np.newaxis, :])**2 + (y_col - cy[np.newaxis, :])**2
    
    # More efficient:
    dx = x[:, np.newaxis] - centers[:, 0][np.newaxis, :]  # (n, 10)
    dy = y[:, np.newaxis] - centers[:, 1][np.newaxis, :]  # (n, 10)
    dist_sq = dx**2 + dy**2
    
    # RBF values: exp(-dist_sq / (2 * sigma^2))
    # Using a common sigma for all
    rbf_values = np.exp(-dist_sq / (2.0 * sigma**2))  # (n, 10)
    
    # Prediction is rbf_values @ params
    # rbf_values: (n, 10)
    # params: (10,)
    pred = rbf_values @ params
    
    return pred
# EVOLVE-BLOCK-END
