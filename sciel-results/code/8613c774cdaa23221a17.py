import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # 9 RBF centers on a 3x3 grid + 1 constant
    centers = np.array([
        [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
        [ 0.0, -1.0], [ 0.0, 0.0], [ 0.0, 1.0],
        [ 1.0, -1.0], [ 1.0, 0.0], [ 1.0, 1.0]
    ])
    
    sigma = 0.7  # RBF width
    
    # Compute squared distances from each point to each center
    # x, y are 1D arrays of length n
    # centers is 9x2
    # dist_sq[i, j] = (x[i] - centers[j,0])^2 + (y[i] - centers[j,1])^2
    
    dx = x[:, np.newaxis] - centers[:, 0][np.newaxis, :]  # n x 9
    dy = y[:, np.newaxis] - centers[:, 1][np.newaxis, :]  # n x 9
    dist_sq = dx**2 + dy**2  # n x 9
    
    # RBF values: exp(-dist_sq / (2 * sigma^2))
    rbf_vals = np.exp(-dist_sq / (2.0 * sigma**2))  # n x 9
    
    # Build the design matrix: [constant, rbf_1, ..., rbf_9]
    # That's n x 10
    design = np.hstack([np.ones((len(x), 1)), rbf_vals])
    
    return design @ params
