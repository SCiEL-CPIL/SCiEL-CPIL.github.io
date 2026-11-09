"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT_MARGIN = 145.0  # seconds, keep a safety margin before 150s kill

def _time_left():
    return _start_time + _TIMEOUT_MARGIN - time.time()

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 parameters that tries to recover an unknown function.
    
    Strategy: Piecewise-Linear (PWL) interpolation over a 3x3 grid of the domain [-1, 1]^2.
    
    This approach partitions the domain into 4 quadrants using the axes x=0 and y=0.
    In each cell, the function is approximated as a bilinear function of (x, y).
    This allows the model to capture local linear trends and discontinuities at the axes,
    which global polynomial bases often struggle with when the underlying function has
    complex local structure.
    
    The 10 parameters are:
    p[0..8]: Values of the function at the 9 vertices of the 3x3 grid:
             (-1, -1), (0, -1), (1, -1)
             (-1,  0), (0,  0), (1,  0)
             (-1,  1), (0,  1), (1,  1)
    p[9]:    Global constant offset.
    
    The prediction is computed via bilinear interpolation within each cell,
    followed by adding the global offset. This is a linear function of the parameters,
    making it well-suited for BFGS optimization.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure inputs are 1D for safety
    x = x.ravel()
    y = y.ravel()
    
    # Determine which side of the axes each point falls on
    x_left = x < 0
    x_right = ~x_left
    y_bottom = y < 0
    y_top = ~y_bottom
    
    # Compute local coordinates within the respective cells
    # For x: if x < 0, local x is in [0, 1] mapped from [-1, 0] -> x + 1
    #        if x >= 0, local x is in [0, 1] mapped from [0, 1] -> x
    x_local = np.where(x_left, x + 1.0, x)
    
    # For y: if y < 0, local y is in [0, 1] mapped from [-1, 0] -> y + 1
    #        if y >= 0, local y is in [0, 1] mapped from [0, 1] -> y
    y_local = np.where(y_bottom, y + 1.0, y)
    
    # Bilinear weights within the local cell
    w00 = (1.0 - x_local) * (1.0 - y_local)  # bottom-left
    w10 = x_local * (1.0 - y_local)          # bottom-right
    w01 = (1.0 - x_local) * y_local          # top-left
    w11 = x_local * y_local                  # top-right
    
    # Initialize weights for the 9 global vertices
    N = len(x)
    weights = np.zeros((N, 9), dtype=np.float64)
    
    # Assign weights to global vertices based on the cell quadrant
    # Global vertex indices:
    # 0: (-1, -1), 1: (0, -1), 2: (1, -1)
    # 3: (-1,  0), 4: (0,  0), 5: (1,  0)
    # 6: (-1,  1), 7: (0,  1), 8: (1,  1)
    
    # Left-Bottom (x < 0, y < 0): Vertices 0, 1, 3, 4
    mask_lb = x_left & y_bottom
    if np.any(mask_lb):
        weights[mask_lb, 0] = w00[mask_lb]
        weights[mask_lb, 1] = w10[mask_lb]
        weights[mask_lb, 3] = w01[mask_lb]
        weights[mask_lb, 4] = w11[mask_lb]
        
    # Right-Bottom (x >= 0, y < 0): Vertices 1, 2, 4, 5
    mask_rb = x_right & y_bottom
    if np.any(mask_rb):
        weights[mask_rb, 1] = w00[mask_rb]
        weights[mask_rb, 2] = w10[mask_rb]
        weights[mask_rb, 4] = w01[mask_rb]
        weights[mask_rb, 5] = w11[mask_rb]
        
    # Left-Top (x < 0, y >= 0): Vertices 3, 4, 6, 7
    mask_lt = x_left & y_top
    if np.any(mask_lt):
        weights[mask_lt, 3] = w00[mask_lt]
        weights[mask_lt, 4] = w10[mask_lt]
        weights[mask_lt, 6] = w01[mask_lt]
        weights[mask_lt, 7] = w11[mask_lt]
        
    # Right-Top (x >= 0, y >= 0): Vertices 4, 5, 7, 8
    mask_rt = x_right & y_top
    if np.any(mask_rt):
        weights[mask_rt, 4] = w00[mask_rt]
        weights[mask_rt, 5] = w10[mask_rt]
        weights[mask_rt, 7] = w01[mask_rt]
        weights[mask_rt, 8] = w11[mask_rt]
        
    # Prediction: weighted sum of vertex values + global offset
    pred = weights @ params[:9] + params[9]
    
    return pred

# EVOLVE-BLOCK-END
