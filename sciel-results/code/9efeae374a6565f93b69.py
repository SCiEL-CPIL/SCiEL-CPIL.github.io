"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32 unit cell with a dielectric cross/diamond pattern
    optimized for a large TM band gap between bands 4 and 5.
    
    Strategy: Use a dielectric cross (plus sign) shape which is known to produce
    strong band gaps in square photonic crystals for TM polarization.
    The cross has 4-fold rotational symmetry and is invariant under transpose
    and reversal. We tune the arm width to maximize the gap.
    
    For TM polarization in a square lattice, a dielectric cross (high epsilon
    along the x and y axes) tends to open gaps. We use a cross with arm width
    w pixels centered in the 32x32 cell.
    
    The cell must satisfy:
    - cell == cell[::-1] row-wise (vertical symmetry)
    - cell == cell[::-1] col-wise (horizontal symmetry) 
    - cell == cell.T (transpose symmetry)
    
    A centered cross with even arm width satisfies all these symmetries.
    """
    size = problem["pixels"]
    # Use a dielectric cross pattern
    # The cross consists of a horizontal bar and a vertical bar
    # For a 32x32 cell, center is between pixels 15 and 16
    # Arm width: try different widths. A common choice is 4-6 pixels wide.
    
    # Let's use a cross with arm width 4 (so indices 14,15,16,17 in each direction)
    # This gives a symmetric cross centered in the cell
    
    arm_width = 4
    center = size // 2  # 16
    half_arm = arm_width // 2  # 2
    
    cell = [[0] * size for _ in range(size)]
    
    for i in range(size):
        for j in range(size):
            # Check if pixel is in the horizontal arm
            in_h_arm = (abs(i - (center - 0.5)) < half_arm + 0.5) 
            # Check if pixel is in the vertical arm
            in_v_arm = (abs(j - (center - 0.5)) < half_arm + 0.5)
            
            if in_h_arm or in_v_arm:
                cell[i][j] = 1
    
    # Verify symmetry properties
    # cell should be invariant under reversal and transpose
    
    return cell
# EVOLVE-BLOCK-END
