"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32 unit cell of a square-lattice photonic crystal
    that maximizes the TM band gap between bands 4 and 5.
    
    Strategy: Use an INVERTED lattice structure.
    Instead of dielectric rods in air (low epsilon background), we use
    air holes in a dielectric background (high epsilon background).
    For TM polarization, the electric field is in the plane and tends to
    concentrate in the low-epsilon (air) regions. This geometry is known
    to produce significant band gaps, potentially larger than the rod
    configuration for certain band orders.
    
    We will use a circular air hole centered in the cell.
    The radius is chosen to be approximately 0.25 lattice constants,
    which is a common optimal range for maximizing gaps in square lattices.
    """
    size = problem["pixels"]
    center = (size - 1) / 2.0
    
    # Define the radius of the air hole in pixels.
    # Lattice constant is 'size' pixels.
    # We try a radius of 0.25 * size.
    radius_frac = 0.25
    radius_px = radius_frac * size
    radius_sq = radius_px * radius_px

    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            dx = i - center
            dy = j - center
            dist_sq = dx * dx + dy * dy
            
            # If inside the hole, it is air (0).
            # Otherwise, it is dielectric (1).
            if dist_sq < radius_sq:
                row.append(0)
            else:
                row.append(1)
        cell.append(row)
        
    return cell
# EVOLVE-BLOCK-END
