import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a separable radial basis function (RBF) representation.
    The output is a weighted sum of 10 fixed 2D Gaussian kernels centered at the vertices
    and midpoints of the unit square, plus linear terms.
    
    Parameters (length 10):
    - params[0:8] : weights for 8 Gaussian kernels (vertices + midpoints)
    - params[8]   : linear coefficient for x
    - params[9]   : linear coefficient for y
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Extract parameters
    weights = params[0:8]
    bx = params[8]
    by = params[9]
    
    # Fixed centers: 4 vertices + 4 midpoints of the unit square [-1,1]x[-1,1]
    # Vertices: (-1,-1), (-1,1), (1,-1), (1,1)
    # Midpoints: (-1,0), (1,0), (0,-1), (0,1)
    cx = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 0.0, 0.0])
    cy = np.array([-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, -1.0, 1.0])
    
    # Fixed width for Gaussian kernels
    sigma = 0.8
    
    # Compute Gaussian kernels
    pred = np.zeros_like(x, dtype=float)
    
    for i in range(8):
        dx = x - cx[i]
        dy = y - cy[i]
        r2 = dx * dx + dy * dy
        kernel = np.exp(-r2 / (2.0 * sigma * sigma))
        pred += weights[i] * kernel
    
    # Add linear terms
    pred += bx * x
    pred += by * y
    
    # Safety: ensure finite output
    if not np.all(np.isfinite(pred)):
        # Fallback to linear-only if anything went wrong
        pred = bx * x + by * y
    
    return pred
