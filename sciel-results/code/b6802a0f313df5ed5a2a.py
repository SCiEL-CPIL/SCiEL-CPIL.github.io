import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Centers: 3x3 grid on [-1, 1]
    # (0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)
    cx = np.array([0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 1.0, -1.0, -1.0])
    cy = np.array([0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    
    # Width parameter gamma
    gamma = 1.5
    
    # Compute squared distances
    # x, y are arrays of length N
    # cx, cy are arrays of length 9
    # We want dist_sq[n, i] = (x[n] - cx[i])^2 + (y[n] - cy[i])^2
    
    dx = x[:, np.newaxis] - cx[np.newaxis, :]
    dy = y[:, np.newaxis] - cy[np.newaxis, :]
    
    dist_sq = dx**2 + dy**2
    
    # Basis functions
    # shape (N, 9)
    basis = np.exp(-gamma * dist_sq)
    
    # Add bias term
    # params[0] is bias, params[1:10] are weights
    bias = params[0]
    weights = params[1:10]
    
    pred = bias + basis @ weights
    
    # Check for non-finite values
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(y)
        
    return pred
