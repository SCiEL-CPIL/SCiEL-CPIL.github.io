"""Combined approach: super-ellipse topology with optimized radius for TM band gap."""
# EVOLVE-BLOCK-START
import math
import time

def design(problem):
    """
    Uses a super-ellipse (squircle) shape for the dielectric pixel distribution
    with an optimized radius to maximize the gap between TM bands 4 and 5.
    
    The super-ellipse with exponent beta=4 aligns with the square lattice symmetry.
    The radius is set to 0.35 lattice constants, which is larger than the seed's
    0.2 and closer to the commonly optimal range for TM band gaps in square
    lattice photonic crystals.
    """
    size = problem["pixels"]
    center = (size - 1) / 2.0  # 15.5 for size 32
    
    # Super-ellipse parameters
    beta = 4.0
    # Radius in normalized units (0 to 0.5)
    # Using 0.35 which is a common optimal radius for TM gaps in square lattices
    radius = 0.35
    
    cell = []
    for i in range(size):
        row = []
        x_norm = (i + 0.5) / size - 0.5
        for j in range(size):
            y_norm = (j + 0.5) / size - 0.5
            # Super-ellipse equation: |x/R|^beta + |y/R|^beta < 1
            val_x = abs(x_norm / radius)
            val_y = abs(y_norm / radius)
            dist = val_x**beta + val_y**beta
            row.append(1 if dist < 1 else 0)
        cell.append(row)
    
    return cell
# EVOLVE-BLOCK-END
