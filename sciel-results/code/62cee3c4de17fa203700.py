import numpy as np
import time

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell.
    Uses a superellipse (rounded square) dielectric rod in air,
    refined to target the 4th-5th TM band gap.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    size = problem["pixels"]
    
    # Start with all air (0s)
    cell = np.zeros((size, size), dtype=int)
    
    # Create coordinates centered at 0
    coords = np.arange(size)
    x = (coords + 0.5) / size - 0.5
    y = (coords + 0.5) / size - 0.5
    
    X, Y = np.meshgrid(x, y)
    
    # Design: Superellipse (rounded square) dielectric rod in air
    # |x|^(2n) + |y|^(2n) <= R^(2n)
    # n_exp = 2.0 is circle, higher is more square
    # For higher-order band gaps, a more square-like shape with appropriate
    # size can enhance the gap.
    
    n_exp = 3.0  # Between circle and square
    R = 0.18     # Radius parameter
    
    if R > 0:
        X_norm = np.abs(X) / R
        Y_norm = np.abs(Y) / R
        
        # Calculate superellipse value
        val = X_norm ** (2 * n_exp) + Y_norm ** (2 * n_exp)
        
        # If val <= 1, it's inside the dielectric
        cell[val <= 1.0] = 1
    
    # Ensure symmetry by construction
    # The superellipse centered at origin is symmetric under 90-degree rotation
    # and reflections about x and y axes
    
    # Check deadline
    if time.time() > deadline:
        pass
        
    return cell.tolist()
