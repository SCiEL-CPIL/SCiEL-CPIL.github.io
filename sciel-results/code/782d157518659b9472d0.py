import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Define 10 RBF centers
    # 9 points on a 3x3 grid in [-1, 1]
    # 1 point at the center with a larger sigma for global trend
    
    centers = np.array([
        [-1.0, -1.0],
        [-1.0,  0.0],
        [-1.0,  1.0],
        [ 0.0, -1.0],
        [ 0.0,  0.0],
        [ 0.0,  1.0],
        [ 1.0, -1.0],
        [ 1.0,  0.0],
        [ 1.0,  1.0],
        [ 0.0,  0.0]  # Duplicate center for different sigma
    ])
    
    # Sigmas for each RBF
    # First 9: local
    # 10th: global
    sigmas = np.array([
        0.7, 0.7, 0.7,
        0.7, 0.7, 0.7,
        0.7, 0.7, 0.7,
        1.5
    ])
    
    # Compute squared distances from each point (x, y) to each center
    # x, y are 1D arrays of length N
    # centers is 10x2
    # We want dist_sq of shape (N, 10)
    
    # Vectorized distance calculation
    # x: (N,), y: (N,)
    # centers: (10, 2)
    
    # Expand dimensions for broadcasting
    # x[:, np.newaxis] is (N, 1)
    # centers[:, 0] is (10,)
    
    dx = x[:, np.newaxis] - centers[:, 0][np.newaxis, :]
    dy = y[:, np.newaxis] - centers[:, 1][np.newaxis, :]
    
    dist_sq = dx**2 + dy**2
    
    # Apply sigmas
    # sigmas: (10,) -> (1, 10) for broadcasting
    sigmas_sq = sigmas**2
    inv_2_sigma_sq = 1.0 / (2.0 * sigmas_sq)
    
    # Basis values: (N, 10)
    basis = np.exp(-dist_sq * inv_2_sigma_sq[np.newaxis, :])
    
    # Predictions: (N, 10) @ (10,) -> (N,)
    return basis @ params
