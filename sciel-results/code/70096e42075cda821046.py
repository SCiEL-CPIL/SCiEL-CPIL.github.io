"""Rewritten program for symbolic regression with a radial basis function (RBF) basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts the target function using a piecewise-linear radial basis function (RBF) basis.
    
    The basis consists of 10 fixed centers in the [0.5, 2]^3 domain. Each basis function
    is a piecewise-linear 'hat' function centered at a grid point, allowing the model
    to capture non-polynomial, smooth, or localized structures that a global polynomial
    basis cannot fit well.
    
    The 10 parameters scale the amplitudes of these basis functions.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if x.size == 0:
        return np.array([], dtype=np.float64)
    
    # Define 10 fixed centers in the [0.5, 2]^3 domain
    # Using a grid-like distribution to cover the space
    centers = np.array([
        [0.5, 0.5, 0.5],
        [2.0, 0.5, 0.5],
        [0.5, 2.0, 0.5],
        [2.0, 2.0, 0.5],
        [0.5, 0.5, 2.0],
        [2.0, 0.5, 2.0],
        [0.5, 2.0, 2.0],
        [2.0, 2.0, 2.0],
        [1.25, 1.25, 0.5],
        [1.25, 1.25, 2.0]
    ], dtype=np.float64)
    
    # Compute piecewise-linear RBF features
    # Each basis function is a product of 1D piecewise-linear hat functions
    # For each center c_i, the basis function is:
    #   phi_i(x,y,z) = hat(x - c_i[0]) * hat(y - c_i[1]) * hat(z - c_i[2])
    # where hat(t) = max(0, 1 - |t|/r) for some radius r
    
    # Choose radius to ensure overlap but locality
    r = 1.5  # Radius for the hat function
    
    n_samples = x.size
    features = np.zeros((n_samples, 10), dtype=np.float64)
    
    for i in range(10):
        cx, cy, cz = centers[i]
        
        # 1D hat functions
        hx = np.maximum(0.0, 1.0 - np.abs(x - cx) / r)
        hy = np.maximum(0.0, 1.0 - np.abs(y - cy) / r)
        hz = np.maximum(0.0, 1.0 - np.abs(z - cz) / r)
        
        features[:, i] = hx * hy * hz
    
    # Add a constant term by modifying the first basis function
    # Actually, let's replace one center with a global constant
    # Better: use 9 RBF centers + 1 constant
    centers[9] = [1.25, 1.25, 1.25]  # Center of the domain
    features[:, 9] = 1.0  # Constant term
    
    # Recompute the 10th feature as a constant
    # Wait, I should keep 10 RBFs or 9 RBFs + 1 constant.
    # Let's use 9 RBFs + 1 constant for better conditioning
    
    # Recompute features with 9 RBFs + 1 constant
    features = np.zeros((n_samples, 10), dtype=np.float64)
    
    for i in range(9):
        cx, cy, cz = centers[i]
        hx = np.maximum(0.0, 1.0 - np.abs(x - cx) / r)
        hy = np.maximum(0.0, 1.0 - np.abs(y - cy) / r)
        hz = np.maximum(0.0, 1.0 - np.abs(z - cz) / r)
        features[:, i] = hx * hy * hz
    
    features[:, 9] = 1.0  # Constant term
    
    # Compute prediction: features (n, 10) dot params (10,) -> (n,)
    pred = features @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
