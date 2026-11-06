"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32 unit cell of a square-lattice photonic crystal
    that maximizes the TM band gap between bands 4 and 5.
    
    Approach: Inverse Structure (Air Holes in Dielectric Background).
    The previous approach used Dielectric Rods in Air (low filling fraction).
    This approach uses a Dielectric Background (high filling fraction) with Air Holes.
    This changes the effective epsilon and band structure significantly,
    potentially opening larger gaps in higher bands.
    
    We create a 32x32 grid filled with 1s (Dielectric, eps=12).
    We then place a symmetric Air Hole (0) in the center.
    We will try a square hole of size 6x6 centered in the cell.
    """
    size = problem["pixels"]
    
    # Initialize with Dielectric (1)
    cell = [[1 for _ in range(size)] for _ in range(size)]
    
    # Define a central air hole
    # Center is at (15.5, 15.5)
    # Let's make a hole of radius 3 pixels (approx)
    # Or a square hole of side 6 (indices 13-18? No, 15.5 center)
    # Indices 13, 14, 15, 16, 17, 18? 
    # Distance from 15.5: 
    # 15.5 - 13 = 2.5
    # 15.5 - 18 = -2.5
    # So 13..18 is 6 pixels wide, centered.
    
    hole_half_width = 3.0 # 6 pixels wide
    
    for i in range(size):
        for j in range(size):
            dx = abs(i - (size - 1) / 2.0)
            dy = abs(j - (size - 1) / 2.0)
            # Square hole
            if dx < hole_half_width and dy < hole_half_width:
                cell[i][j] = 0
            # Optional: Add small holes at corners to enhance symmetry effects?
            # For now, keep it simple: Central hole only.
            
    return cell
# EVOLVE-BLOCK-END
