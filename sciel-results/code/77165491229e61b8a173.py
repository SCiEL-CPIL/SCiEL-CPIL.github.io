"""
Symbolic regression using a Radial Basis Function (RBF) network.
This program uses a fixed set of RBF centers and widths.
The 10 parameters are the weights for each RBF center.
This creates a linear-in-parameters model that BFGS can fit efficiently.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts the target value using a Radial Basis Function (RBF) network.
    
    Args:
        x: Input feature x (array of shape (n,))
        y: Input feature y (array of shape (n,))
        params: Weights for the 10 RBF centers (array of shape (10,))
        
    Returns:
        Prediction array of shape (n,)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.size
    
    # Ensure params is a flat array of at least length 10
    params = np.asarray(params, dtype=np.float64).ravel()
    if params.size < 10:
        params = np.concatenate([params, np.zeros(10 - params.size)])
    weights = params[:10]
    
    # Define fixed RBF centers in the [-1, 1] x [-1, 1] domain.
    # We use a 3x3 grid plus the origin to ensure coverage.
    # Total 10 centers to match the 10 parameters.
    centers = np.array([
        [-1.0, -1.0], [ 0.0, -1.0], [ 1.0, -1.0],
        [-1.0,  0.0], [ 0.0,  0.0], [ 1.0,  0.0],
        [-1.0,  1.0], [ 0.0,  1.0], [ 1.0,  1.0],
        [ 0.0,  0.0] # Duplicate origin? No, let's pick a different one or adjust.
                     # Actually, a 3x3 grid is 9 points. We need 10.
                     # Let's add a point at (0.5, 0.5) or similar to break symmetry?
                     # Or just use a 4x3 grid? No, let's stick to 3x3 + one extra.
                     # Let's use a 3x3 grid and one extra at (0.5, 0.5)
    ])
    
    # Correcting centers to be distinct and well-distributed
    centers = np.array([
        [-1.0, -1.0], [ 0.0, -1.0], [ 1.0, -1.0],
        [-1.0,  0.0], [ 0.0,  0.0], [ 1.0,  0.0],
        [-1.0,  1.0], [ 0.0,  1.0], [ 1.0,  1.0],
        [ 0.5,  0.5]
    ])
    
    # Define fixed widths (sigma) for each RBF.
    # A mix of wide and narrow kernels helps capture global and local structures.
    # We need 10 widths.
    widths = np.array([
        1.0, 1.0, 1.0,   # Corners: wide
        0.7, 0.7, 0.7,   # Edges: medium
        1.0, 1.0, 1.0,   # Corners: wide
        0.5              # Center-ish: narrow for detail
    ])
    
    # Compute the RBF features for each point.
    # X is (n, 2), centers is (10, 2)
    # We want dist_sq[i, j] = ||x_i - center_j||^2
    # Using broadcasting: (n, 1, 2) - (1, 10, 2) -> (n, 10, 2)
    # This creates a large intermediate array (n*10*2). For n=200, this is 4000 floats, which is fine.
    # However, to be safer and potentially faster, we can compute in chunks or use a loop.
    # Given n is small (200), broadcasting is efficient enough in NumPy.
    
    x_col = x[:, np.newaxis] # (n, 1)
    y_col = y[:, np.newaxis] # (n, 1)
    
    cx = centers[:, 0] # (10,)
    cy = centers[:, 1] # (10,)
    
    # Squared distances: (n, 1) - (1, 10) -> (n, 10)
    dx = x_col - cx[np.newaxis, :]
    dy = y_col - cy[np.newaxis, :]
    dist_sq = dx * dx + dy * dy
    
    # Apply Gaussian kernel: exp(-dist_sq / (2 * width^2))
    # widths is (10,), so we need to broadcast it.
    # (10,) -> (1, 10)
    w_sq_2 = 2.0 * (widths * widths)[np.newaxis, :]
    
    # Avoid division by zero or very small numbers, though widths are fixed and non-zero
    features = np.exp(-dist_sq / w_sq_2)
    
    # Prediction is the dot product of features (n, 10) and weights (10,)
    pred = features @ weights
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
