"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def design(problem):
    """
    Design a 2x2 supercell of dielectric rods in air.
    This increases the complexity of the unit cell, potentially opening a larger gap
    at higher band indices (4-5) compared to the primitive cell.
    """
    size = problem["pixels"]
    cell = [[0 for _ in range(size)] for _ in range(size)]
    
    # Define centers for a 2x2 supercell of rods
    # The 32x32 cell is divided into four 16x16 quadrants.
    # Centers are at the middle of each quadrant.
    # Indices are 0 to 31.
    # Quadrant centers:
    # Top-Left: (7.5, 7.5)
    # Top-Right: (7.5, 23.5)
    # Bottom-Left: (23.5, 7.5)
    # Bottom-Right: (23.5, 23.5)
    
    centers = [
        (7.5, 7.5),
        (7.5, 23.5),
        (23.5, 7.5),
        (23.5, 23.5)
    ]
    
    # Radius of the rods.
    # The effective lattice constant for the rods is 16 pixels.
    # Optimal radius for square lattice is approx 0.2-0.3 * a.
    # Let's try 0.25 * 16 = 4.0 pixels.
    radius = 4.0
    
    for i in range(size):
        for j in range(size):
            # Pixel center coordinates
            x = i + 0.5
            y = j + 0.5
            
            is_dielectric = 0
            for cx, cy in centers:
                dist_sq = (x - cx)**2 + (y - cy)**2
                if dist_sq < radius**2:
                    is_dielectric = 1
                    break
            
            cell[i][j] = is_dielectric
            
    return cell
# EVOLVE-BLOCK-END
