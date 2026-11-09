import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a Gaussian Radial Basis Function (RBF) basis.
    
    We use 10 fixed centers distributed across the domain [0.5, 2]^3.
    The basis functions are Gaussian kernels with a fixed width.
    This provides a flexible, non-polynomial model that can capture
    complex non-linear dependencies better than low-degree polynomials.
    """
    # Fixed centers
    # 8 corners
    centers = np.array([
        [0.5, 0.5, 0.5],
        [2.0, 0.5, 0.5],
        [0.5, 2.0, 0.5],
        [0.5, 0.5, 2.0],
        [2.0, 2.0, 0.5],
        [2.0, 0.5, 2.0],
        [0.5, 2.0, 2.0],
        [2.0, 2.0, 2.0],
        # 2 internal points
        [1.25, 1.25, 1.25],
        [1.25, 0.5, 1.25]
    ])
    
    # Fixed width
    sigma = 0.6
    
    # Compute basis functions
    # x, y, z are arrays of shape (N,)
    # We need to compute distance from each point (x_i, y_i, z_i) to each center c_j
    # Resulting basis matrix is (N, 10)
    
    # To avoid creating large intermediate arrays, we can loop over centers or vectorize carefully.
    # Vectorized approach:
    # X = np.column_stack((x, y, z)) # Shape (N, 3)
    # D = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2) # Shape (N, 10)
    # Basis = np.exp(-D**2 / (2 * sigma**2))
    
    # This creates an array of size N*10*3, which is small for N=200.
    X = np.column_stack((x, y, z))
    # Reshape for broadcasting: (N, 1, 3) - (1, 10, 3)
    D2 = np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
    basis = np.exp(-D2 / (2.0 * sigma * sigma))
    
    # Linear combination
    # params is shape (10,)
    pred = basis @ params
    
    return pred
