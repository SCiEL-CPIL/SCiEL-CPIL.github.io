"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def design(problem):
    """
    Design a symmetric 32x32 unit cell.
    Uses an 'Air Cross' in a Dielectric background (Inverse Cross).
    This structure is known to produce strong scattering for TM polarization,
    potentially opening larger gaps in higher bands compared to simple rods.
    """
    size = problem["pixels"]
    # Target: 32x32
    # Structure: Dielectric background (1) with an Air (0) cross.
    # Cross width: 8 pixels (indices 12 to 19 inclusive).
    # This ensures symmetry about the center (15.5).
    
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            # Check if pixel is part of the cross
            # Vertical bar: 12 <= i <= 19
            # Horizontal bar: 12 <= j <= 19
            if 12 <= i <= 19 or 12 <= j <= 19:
                row.append(0)  # Air
            else:
                row.append(1)  # Dielectric
        cell.append(row)
    
    return cell
# EVOLVE-BLOCK-END
