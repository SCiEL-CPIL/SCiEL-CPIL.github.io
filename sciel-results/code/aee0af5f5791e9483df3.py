# EVOLVE-BLOCK-START
import numpy as np


def squares_in_square10() -> tuple:
    """
    Place 10 squares in the unit square to maximize the sum of their side lengths.
    
    This implementation uses a known efficient packing:
    - 2 large squares of side 0.5 placed side-by-side at the bottom.
    - 8 small squares of side 0.25 arranged in a 2x4 grid in the top half.
    
    Sum of side lengths = 2 * 0.5 + 8 * 0.25 = 1.0 + 2.0 = 3.0.
    This is an improvement over the baseline grid packing (sum 2.5).
    """
    n = 10
    centers = np.zeros((n, 2), dtype=np.float64)
    sides = np.zeros(n, dtype=np.float64)
    
    # Large squares (indices 0, 1)
    side_large = 0.5
    # Square 0: bottom-left
    centers[0] = [0.25, 0.25]
    sides[0] = side_large
    # Square 1: bottom-right
    centers[1] = [0.75, 0.25]
    sides[1] = side_large
    
    # Small squares (indices 2-9)
    side_small = 0.25
    # Top half is y in [0.5, 1.0]
    # 2 rows, 4 columns
    # Row 0 (bottom of top half): y center = 0.5 + 0.25/2 = 0.625
    # Row 1 (top of top half): y center = 0.75 + 0.25/2 = 0.875
    # Columns: x centers = 0.125, 0.375, 0.625, 0.875
    
    x_centers = [0.125, 0.375, 0.625, 0.875]
    y_centers = [0.625, 0.875]
    
    idx = 2
    for y_c in y_centers:
        for x_c in x_centers:
            centers[idx] = [x_c, y_c]
            sides[idx] = side_small
            idx += 1
            
    return centers, sides


# EVOLVE-BLOCK-END
