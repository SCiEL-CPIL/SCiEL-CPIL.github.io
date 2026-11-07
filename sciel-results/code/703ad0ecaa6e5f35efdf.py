import numpy as np

def equation(x, y, params):
    # Use a sparse linear combination of 10 Radial Basis Functions (RBFs)
    # with fixed centers and widths over [-1, 1]^2.
    # Centers are chosen from a Halton-like grid or deterministic set.
    # Widths are fixed to avoid optimizing geometric parameters.
    
    # Define 10 fixed centers in [-1, 1]^2
    # Using a quasi-random or grid-based approach for good coverage
    centers = np.array([
        [-0.8, -0.8],
        [ 0.8, -0.8],
        [-0.8,  0.8],
        [ 0.8,  0.8],
        [ 0.0,  0.0],
        [-0.5,  0.5],
        [ 0.5, -0.5],
        [-0.5, -0.5],
        [ 0.5,  0.5],
        [ 0.0,  0.0]  # duplicate or adjust for better spread
    ])
    
    # Better centers: use a 3x3 grid plus center, or Halton sequence
    # Let's use a deterministic set of 10 points that cover the domain well
    # Halton sequence in 2D for first 10 points
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    centers = np.array([[halton(i+1, 2), halton(i+1, 3)] for i in range(10)])
    # Map from [0,1] to [-1,1]
    centers = 2.0 * centers - 1.0
    
    # Fixed widths (sigma) for Gaussian RBFs
    # Use a moderate width to allow local features
    sigma = 0.5
    
    # Compute RBFs: exp(-||x - c_i||^2 / (2 * sigma^2))
    # x and y are arrays of shape (N,)
    # We need to compute distance from each point (x_i, y_i) to each center (cx_j, cy_j)
    
    # Reshape x and y for broadcasting
    x_col = x.reshape(-1, 1)
    y_col = y.reshape(-1, 1)
    
    # Centers shape: (10, 2)
    # Compute squared distances
    # dx = x_col - centers[:, 0].reshape(1, -1)
    # dy = y_col - centers[:, 1].reshape(1, -1)
    # dist_sq = dx**2 + dy**2
    # rbf = np.exp(-dist_sq / (2 * sigma**2))
    
    dx = x_col - centers[:, 0]
    dy = y_col - centers[:, 1]
    dist_sq = dx**2 + dy**2
    rbf = np.exp(-dist_sq / (2 * sigma**2))
    
    # rbf shape: (N, 10)
    # params shape: (10,)
    # Prediction = rbf @ params
    
    return rbf @ params
