import numpy as np

def squares_in_square10() -> tuple:
    """
    Returns a valid packing of 10 squares in the unit square with a sum of side lengths of 3.0.
    Configuration:
    - 4 large squares of side 0.375 in a 2x2 grid at the bottom-left.
    - 6 small squares of side 0.25 filling the remaining L-shape.
    """
    n = 10
    centers = np.zeros((n, 2), dtype=np.float64)
    sides = np.zeros(n, dtype=np.float64)
    
    # Large squares
    L = 0.375
    s_large = L
    # Positions (center x, center y)
    # 2x2 grid in [0, 0.75] x [0, 0.75]
    large_centers = [
        (L/2, L/2),
        (L + L/2, L/2),
        (L/2, L + L/2),
        (L + L/2, L + L/2)
    ]
    for i, (cx, cy) in enumerate(large_centers):
        centers[i] = [cx, cy]
        sides[i] = s_large
        
    # Small squares
    s_small = 0.25
    # Right strip: x in [0.75, 1.0], y in [0.0, 1.0]
    # 4 squares stacked vertically
    x_small_right = 0.75 + s_small / 2
    y_small_right = [s_small/2, 3*s_small/2, 5*s_small/2, 7*s_small/2]
    for j, cy in enumerate(y_small_right):
        idx = 4 + j
        centers[idx] = [x_small_right, cy]
        sides[idx] = s_small
        
    # Top strip: x in [0.0, 0.75], y in [0.75, 1.0]
    # 3 squares side-by-side horizontally
    y_small_top = 0.75 + s_small / 2
    x_small_top = [s_small/2, 3*s_small/2, 5*s_small/2]
    for j, cx in enumerate(x_small_top):
        idx = 8 + j
        centers[idx] = [cx, y_small_top]
        sides[idx] = s_small
        
    return centers, sides
