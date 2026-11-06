import numpy as np
import time
import math

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell.
    Uses a ring-based geometry in air background to target higher band gaps (4th-5th TM).
    Ring geometries are known to produce larger band gaps in square photonic crystals.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    size = problem["pixels"]
    
    # Start with all air (0s)
    cell = np.zeros((size, size), dtype=int)
    
    # Create coordinates centered at 0
    coords = np.arange(size)
    x = (coords + 0.5) / size - 0.5
    y = (coords + 0.5) / size - 0.5
    
    X, Y = np.meshgrid(x, y)
    
    # Design: Dielectric ring (annulus) in air background
    # This creates strong Bragg scattering for higher-order bands
    # Ring parameters: inner radius r_in, outer radius r_out
    # Centered at origin
    
    # Try a ring with moderate thickness
    r_in = 0.15
    r_out = 0.35
    
    R = np.sqrt(X**2 + Y**2)
    
    # Fill the ring region with dielectric
    mask = (R >= r_in) & (R <= r_out)
    cell[mask] = 1
    
    # Ensure symmetry by construction (circle is symmetric)
    
    # Check deadline
    if time.time() > deadline:
        pass
        
    return cell.tolist()
