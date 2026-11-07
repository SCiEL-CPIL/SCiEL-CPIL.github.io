import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF Basis with fixed centers.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Define 10 centers in the domain [0.5, 2]
    # Spread them out to cover the space
    centers = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.5],
        [1.0, 1.5, 1.0],
        [1.5, 1.0, 1.0],
        [1.5, 1.5, 1.5],
        [1.0, 1.5, 1.5],
        [1.5, 1.0, 1.5],
        [1.5, 1.5, 1.0],
        [1.0, 1.0, 2.0],
        [2.0, 1.0, 1.0]
    ])
    
    sigma = 0.6  # Width of the RBFs
    
    # Calculate distances
    # x, y, z are arrays of shape (N,)
    # centers are (10, 3)
    # We need (N, 10) distances
    
    # Reshape x, y, z to (N, 1)
    X = x.reshape(-1, 1)
    Y = y.reshape(-1, 1)
    Z = z.reshape(-1, 1)
    
    # Centers reshaped to (1, 10, 3) for broadcasting?
    # Or loop. Loop is safe for small N? N=200. 10 centers.
    # Vectorized is better.
    
    # dx = X - centers[:, 0] -> (N, 10)
    # dy = Y - centers[:, 1] -> (N, 10)
    # dz = Z - centers[:, 2] -> (N, 10)
    
    cx = centers[:, 0]
    cy = centers[:, 1]
    cz = centers[:, 2]
    
    dx = X - cx  # (N, 10)
    dy = Y - cy  # (N, 10)
    dz = Z - cz  # (N, 10)
    
    dist_sq = dx**2 + dy**2 + dz**2
    phi = np.exp(-dist_sq / (2 * sigma**2))
    
    # result = phi @ p
    result = phi @ p
    
    return result.flatten()
