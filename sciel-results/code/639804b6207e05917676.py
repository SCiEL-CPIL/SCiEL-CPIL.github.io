"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def design(problem):
    """
    Design a 32x32 unit cell for a 2D square-lattice photonic crystal.
    Uses a D4-symmetric design with a central dielectric cross and
    four corner dielectric patches to maximize the TM band gap
    between the 4th and 5th bands.
    
    The design uses a "checkerboard-like" pattern with additional
    connectivity to create strong Bragg scattering at higher orders.
    """
    start_time = time.time()
    deadline = 190  # Safety margin before 230s kill
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Initialize with air (0)
    cell = [[0 for _ in range(size)] for _ in range(size)]
    
    # Strategy: Use a design that has strong 4-fold symmetry and
    # creates a large gap between TM bands 4 and 5.
    # 
    # Based on literature, good designs for higher-order TM gaps in
    # square lattices often involve:
    # 1. A central cross or plus shape
    # 2. Four corner patches
    # 3. Sometimes additional intermediate features
    #
    # Let's try a design with:
    # - A thick central cross
    # - Four corner squares
    # - The pattern should be D4 symmetric
    
    # Define the central cross
    # The cross should be centered at (15.5, 15.5) in pixel coordinates
    # Let's make the cross arms have a certain width
    
    # Parameter: width of cross arms (in pixels)
    cross_width = 6
    # Parameter: size of corner patches
    corner_size = 5
    
    # Create the cross
    center = size // 2  # 16
    half_cross = cross_width // 2
    
    for i in range(size):
        for j in range(size):
            # Check if pixel is in the vertical arm of the cross
            if abs(j - (center - 0.5)) < half_cross:
                cell[i][j] = 1
            # Check if pixel is in the horizontal arm of the cross
            elif abs(i - (center - 0.5)) < half_cross:
                cell[i][j] = 1
    
    # Add corner patches
    # Corners are at (0,0), (0,31), (31,0), (31,31)
    # But we need to be careful about symmetry
    # Let's add patches near the corners
    
    # Top-left corner
    for i in range(corner_size):
        for j in range(corner_size):
            cell[i][j] = 1
    
    # Top-right corner
    for i in range(corner_size):
        for j in range(size - corner_size, size):
            cell[i][j] = 1
    
    # Bottom-left corner
    for i in range(size - corner_size, size):
        for j in range(corner_size):
            cell[i][j] = 1
    
    # Bottom-right corner
    for i in range(size - corner_size, size):
        for j in range(size - corner_size, size):
            cell[i][j] = 1
    
    # Verify D4 symmetry
    # 1. cell == cell[::-1] row-wise (vertical flip)
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[size - 1 - i][j]:
                # Fix by averaging (but since binary, we need to decide)
                # For now, just ensure symmetry by copying
                cell[size - 1 - i][j] = cell[i][j]
    
    # 2. cell == cell[::-1] column-wise (horizontal flip)
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[i][size - 1 - j]:
                cell[i][size - 1 - j] = cell[i][j]
    
    # 3. cell == its transpose
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[j][i]:
                cell[j][i] = cell[i][j]
    
    # Re-verify and fix any remaining asymmetry
    # Vertical flip
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[size - 1 - i][j]:
                # Keep the value from the top half
                cell[size - 1 - i][j] = cell[i][j]
    
    # Horizontal flip
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[i][size - 1 - j]:
                # Keep the value from the left half
                cell[i][size - 1 - j] = cell[i][j]
    
    # Transpose
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[j][i]:
                # Keep the value from the upper triangle
                cell[j][i] = cell[i][j]
    
    return cell
# EVOLVE-BLOCK-END
