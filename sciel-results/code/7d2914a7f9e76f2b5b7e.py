import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Structural change: Radial Basis Function (RBF) network with fixed centers
    # Approximating a Gaussian Process with fixed hyperparameters.
    # 10 centers in [0.5, 2]^3.
    
    # Fixed centers
    centers = np.array([
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 2.0],
        [0.5, 2.0, 0.5],
        [0.5, 2.0, 2.0],
        [2.0, 0.5, 0.5],
        [2.0, 0.5, 2.0],
        [2.0, 2.0, 0.5],
        [2.0, 2.0, 2.0],
        [1.25, 1.25, 1.25],
        [1.25, 1.25, 0.5]
    ])
    
    # Length scale
    l = 1.0
    l2 = 2.0 * l * l
    
    # Compute distances
    # x, y, z are shape (N,)
    # centers is shape (10, 3)
    # We want distances from each point (x[i], y[i], z[i]) to each center c[j]
    
    # Efficient calculation:
    # dx = x[:, None] - centers[None, :, 0]
    # dy = y[:, None] - centers[None, :, 1]
    # dz = z[:, None] - centers[None, :, 2]
    # dist2 = dx**2 + dy**2 + dz**2
    
    dx = x[:, None] - centers[None, :, 0]
    dy = y[:, None] - centers[None, :, 1]
    dz = z[:, None] - centers[None, :, 2]
    
    dist2 = dx**2 + dy**2 + dz**2
    
    # Gaussian kernel
    K = np.exp(-dist2 / l2)
    
    # Result: K (N, 10) @ p (10,) -> (N,)
    result = K @ p
    
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    return result
