# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Radial Basis Function (RBF) approximation.
    
    The model is a linear combination of 9 Gaussian RBFs centered on a 3x3x3 grid
    within the input domain [0.5, 2]^3, plus a bias term.
    
    params:
      0-8: Weights for the 9 RBF centers.
      9:   Bias term.
    """
    # Input arrays are expected to be 1D or broadcastable. 
    # The evaluator passes 200 points, likely as 1D arrays of length 200.
    
    # Define centers on a 3x3x3 grid in [0.5, 2]
    # Values: 0.5, 1.25, 2.0
    cx = np.array([0.5, 1.25, 2.0])
    cy = np.array([0.5, 1.25, 2.0])
    cz = np.array([0.5, 1.25, 2.0])
    
    # Create meshgrid of centers
    # We need to compute distance from each input point (x[i], y[i], z[i]) to each center (cx[j], cy[k], cz[l])
    # It is more efficient to compute this in a vectorized manner.
    
    # x, y, z are shape (N,)
    # Centers are shape (9,)
    
    # Reshape inputs to (N, 1) for broadcasting
    x_col = x[:, np.newaxis]
    y_col = y[:, np.newaxis]
    z_col = z[:, np.newaxis]
    
    # Centers flattened: 9 centers
    # c_x, c_y, c_z are shape (3,)
    # We want a matrix of distances of shape (N, 9)
    
    # dx shape (N, 3), dy shape (N, 3), dz shape (N, 3)
    dx = x_col - cx[np.newaxis, :]
    dy = y_col - cy[np.newaxis, :]
    dz = z_col - cz[np.newaxis, :]
    
    # We need to combine these 3D coordinates for 9 centers.
    # Center indices:
    # 0: (cx[0], cy[0], cz[0])
    # 1: (cx[0], cy[0], cz[1])
    # ...
    # This is a Cartesian product.
    
    # Let's construct the full center coordinates array C of shape (9, 3)
    C = np.array(np.meshgrid(cx, cy, cz)).T.reshape(-1, 3)
    
    # C is shape (9, 3)
    # Inputs are shape (N, 3)
    
    # Compute squared distances: ||x_i - c_j||^2
    # x_data shape (N, 3)
    x_data = np.column_stack((x, y, z))
    
    # dist_sq shape (N, 9)
    # Using broadcasting: x_data[:, np.newaxis, :] - C[np.newaxis, :, :]
    diff = x_data[:, np.newaxis, :] - C[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # RBF width sigma
    # Domain size is 1.5. Grid spacing is 0.75.
    # Sigma = 0.4 provides good overlap.
    sigma_sq = 0.16 # 0.4^2
    two_sigma_sq = 2.0 * sigma_sq
    
    # Compute basis functions
    # phi = exp(-dist_sq / (2 * sigma^2))
    # Clip dist_sq to prevent underflow issues, though exp handles large negatives fine.
    # Max dist_sq is roughly 3 * 1.5^2 = 6.75. 6.75 / 0.32 ~ 21. exp(-21) is ~ 7e-10. Safe.
    basis = np.exp(-dist_sq / two_sigma_sq)
    
    # basis shape (N, 9)
    # params[0:9] are weights for these basis functions.
    # params[9] is the bias.
    
    weights = params[0:9]
    bias = params[9]
    
    # Linear combination
    pred = basis @ weights + bias
    
    return pred
# EVOLVE-BLOCK-END
