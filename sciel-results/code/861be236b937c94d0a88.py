"""Design a symmetric 32x32 photonic crystal unit cell for maximum TM band gap (bands 4-5)."""
import math

def design(problem):
    """
    Design a symmetric 32x32 unit cell.
    Uses a 2x2 square lattice of dielectric rods (squares or circles) in air.
    This is a distinct approach from a single central blob, leveraging
    the periodicity of a superlattice within the unit cell to open gaps.
    """
    size = problem["pixels"]
    cell = [[0] * size for _ in range(size)]
    
    # Define centers for a 2x2 array of rods
    # Symmetric about center (15.5, 15.5)
    # Centers at (8, 8), (8, 23), (23, 8), (23, 23)
    centers = [(8, 8), (8, 23), (23, 8), (23, 23)]
    
    # Radius of the rods. 
    # A radius of 3.5 creates a 7x7 square-like footprint if using Chebyshev, 
    # or a circle. Let's use a circular radius for better symmetry properties 
    # in the Fourier domain, though square rods are also common.
    # Let's try a radius that gives ~30-40% fill factor.
    # Area of 4 circles: 4 * pi * r^2. Total area 1024.
    # If r=3, area = 4 * 28.27 = 113. Fill ~11%. Too low.
    # If r=5, area = 4 * 78.5 = 314. Fill ~30%. Good.
    # If r=6, area = 4 * 113 = 452. Fill ~44%. Good.
    
    # Let's try r=5.5
    radius = 5.5
    
    for i in range(size):
        for j in range(size):
            for cx, cy in centers:
                dist_sq = (i - cx)**2 + (j - cy)**2
                if dist_sq <= radius**2:
                    cell[i][j] = 1
                    break
    
    return cell
