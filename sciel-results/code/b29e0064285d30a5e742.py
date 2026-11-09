import numpy as np
from scipy.optimize import minimize

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Radial basis function (RBF) representation
    # Use 10 fixed 2D Gaussian kernels centered at vertices and midpoints of the unit square
    # This allows local weights to capture arbitrary curvature without periodicity constraints
    
    # Define kernel centers: 4 corners + 4 edge midpoints + 1 center + 1 extra
    centers = np.array([
        [-1.0, -1.0],  # bottom-left
        [1.0, -1.0],   # bottom-right
        [-1.0, 1.0],   # top-left
        [1.0, 1.0],    # top-right
        [0.0, -1.0],   # bottom-mid
        [0.0, 1.0],    # top-mid
        [-1.0, 0.0],   # left-mid
        [1.0, 0.0],    # right-mid
        [0.0, 0.0],    # center
        [0.5, 0.5]     # extra point for better coverage
    ])
    
    # Width parameter for Gaussian kernels - needs to be tuned
    # Smaller width = more localized, larger width = smoother
    sigma = 0.8
    
    # Compute Gaussian RBF values
    # For each data point (x[i], y[i]), compute distance to each center
    # RBF_j(x, y) = exp(-||[x,y] - center_j||^2 / (2 * sigma^2))
    
    # Stack x and y into 2D points
    points = np.column_stack((x, y))  # shape: (n, 2)
    
    # Compute squared distances from each point to each center
    # points: (n, 2), centers: (10, 2)
    # distances: (n, 10)
    
    # Use broadcasting: points[:, np.newaxis, :] - centers[np.newaxis, :, :]
    diffs = points[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (n, 10, 2)
    sq_dists = np.sum(diffs ** 2, axis=2)  # (n, 10)
    
    # Compute RBF values
    rbf_values = np.exp(-sq_dists / (2.0 * sigma ** 2))  # (n, 10)
    
    # Linear combination with params
    # rbf_values: (n, 10), params: (10,)
    predictions = rbf_values @ params
    
    return predictions
