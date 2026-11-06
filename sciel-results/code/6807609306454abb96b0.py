import numpy as np
import time

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell.
    Uses air background with 4 dielectric rods in a 2x2 arrangement to
    target higher band gaps (4th-5th TM).
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
    
    # Design: 4 dielectric rods in an air background
    # Positions: (±dx, ±dy)
    # This creates a 2x2 supercell-like structure which often enhances
    # higher-order band gaps compared to a single rod.
    
    dx = 0.25
    dy = 0.25
    R_rod = 0.15
    
    # Rod at (dx, dy)
    dist1 = np.sqrt((X - dx)**2 + (Y - dy)**2)
    cell[dist1 <= R_rod] = 1
    
    # Rod at (-dx, dy)
    dist2 = np.sqrt((X + dx)**2 + (Y - dy)**2)
    cell[dist2 <= R_rod] = 1
    
    # Rod at (dx, -dy)
    dist3 = np.sqrt((X - dx)**2 + (Y + dy)**2)
    cell[dist3 <= R_rod] = 1
    
    # Rod at (-dx, -dy)
    dist4 = np.sqrt((X + dx)**2 + (Y + dy)**2)
    cell[dist4 <= R_rod] = 1
    
    # Ensure symmetry by construction
    # The pattern is symmetric under 90-degree rotation and reflections
    
    # Check deadline
    if time.time() > deadline:
        pass
        
    return cell.tolist()
