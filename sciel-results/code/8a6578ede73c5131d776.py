"""Improved program using a super-ellipse topology to maximize TM band gap."""
# EVOLVE-BLOCK-START
import math
import time

def design(problem):
    """
    Uses a super-ellipse (squircle) shape for the dielectric pixel distribution.
    Optimized for a larger fill fraction (~35%) which is often optimal for
    TM band gaps in square lattice photonic crystals.
    """
    start_time = time.time()
    size = problem["pixels"]
    
    # Parameters for the super-ellipse
    # Beta = 4 (Squircle) is a good balance between circle and square symmetry.
    # We aim for a fill fraction of ~0.35 to maximize the gap.
    # Area of superellipse = C * r^2.
    # For beta=4, C ~ 3.708.
    # Target Area = 0.35.
    # r = sqrt(0.35 / 3.708) ~ 0.307.
    
    beta = 4.0
    area_coeff = 3.708
    target_area = 0.35
    r_beta = math.sqrt(target_area / area_coeff)
    
    # We can try a few radii if time permits, but a single good guess is safer.
    # Let's stick to one optimized radius.
    
    cell = []
    for i in range(size):
        row = []
        x_norm = (i + 0.5) / size - 0.5
        for j in range(size):
            y_norm = (j + 0.5) / size - 0.5
            # Super-ellipse equation: |x/R|^beta + |y/R|^beta < 1
            val_x = abs(x_norm / r_beta)
            val_y = abs(y_norm / r_beta)
            dist = val_x**beta + val_y**beta
            row.append(1 if dist < 1 else 0)
        cell.append(row)
        
        # Time check (though generation is fast, just in case)
        if time.time() - start_time > 110:
            break

    return cell
# EVOLVE-BLOCK-END
