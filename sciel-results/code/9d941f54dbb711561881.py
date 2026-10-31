"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Hexagonal packing with 5 rows: 5, 4, 5, 4, 3 circles
    # This arrangement has a higher packing density than the square grid.
    
    # Calculate optimal radius r for perimeter 4 (W + H = 2)
    # Width W is determined by the widest row (5 circles): W = 10r
    # Height H is determined by 5 rows: H = 2r + 4 * r * sqrt(3)
    # Constraint: W + H = 2
    # 10r + 2r + 4r*sqrt(3) = 2
    # r * (12 + 4*sqrt(3)) = 2
    # r = 1 / (6 + 2*sqrt(3))
    
    r = 1.0 / (6.0 + 2.0 * np.sqrt(3.0))
    
    W = 10.0 * r
    H = 2.0 * r + 4.0 * r * np.sqrt(3.0)
    
    # Verify perimeter is 4
    # 2 * (W + H) = 2 * 2 = 4. Correct.
    
    centers = []
    rows = [5, 4, 5, 4, 3]
    dx = 2.0 * r
    dy = r * np.sqrt(3.0)
    
    for i, count in enumerate(rows):
        y = r + i * dy
        # Center each row horizontally within W
        # Width of row = (count - 1) * dx
        # Start x = r + (W - (count - 1) * dx) / 2.0
        start_x = r + (W - (count - 1) * dx) / 2.0
        for j in range(count):
            x = start_x + j * dx
            centers.append([x, y, r])
            
    return np.array(centers)
# EVOLVE-BLOCK-END
