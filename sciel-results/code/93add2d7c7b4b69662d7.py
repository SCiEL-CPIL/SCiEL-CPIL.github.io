import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 115.0 # Safety margin before 120s

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check time limit
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Return a simple fallback if we are running out of time
        # This shouldn't happen for this simple calculation
        return np.zeros_like(x)

    # Fixed centers for RBFs: 3x3 grid
    xs = np.array([-1.0, 0.0, 1.0])
    ys = np.array([-1.0, 0.0, 1.0])
    
    # Create meshgrid of centers
    cx, cy = np.meshgrid(xs, ys, indexing='ij')
    cx = cx.flatten()
    cy = cy.flatten()
    
    # Number of RBFs
    n_rbf = 9
    
    # Extract weights and bias
    w = params[:n_rbf]
    b = params[n_rbf]
    
    # Compute distances from (x, y) to each center
    # x, y are (N,)
    # cx, cy are (9,)
    # We want (N, 9) distances
    dx = x[:, np.newaxis] - cx[np.newaxis, :]
    dy = y[:, np.newaxis] - cy[np.newaxis, :]
    
    # Squared distances
    dist_sq = dx**2 + dy**2
    
    # RBF values: exp(-gamma * dist_sq)
    gamma = 2.0
    rbf_vals = np.exp(-gamma * dist_sq)
    
    # Prediction: sum(w_i * rbf_i) + b
    pred = rbf_vals @ w + b
    
    return pred
