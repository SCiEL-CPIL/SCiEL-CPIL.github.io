# EVOLVE-BLOCK-START
import numpy as np


def squares_in_square10() -> tuple:
    """
    Packs 10 squares into the unit square to maximize the sum of their side lengths.
    
    Strategy:
    Uses a shelf-based packing model.
    We define 4 horizontal shelves.
    Shelf 1: 1 square of size s1.
    Shelf 2: 3 squares of size s2.
    Shelf 3: 3 squares of size s3.
    Shelf 4: 3 squares of size s4.
    
    To maximize the sum of sides, we set s2=s3=s4=s_large and s1=s_small.
    Constraints:
    - Width: 3*s_large <= 1 => s_large <= 1/3.
    - Height: s_small + 3*s_large <= 1.
    
    Optimization:
    s_small = 1 - 3*s_large.
    Objective: 9*s_large + s_small = 6*s_large + 1.
    Maximized when s_large is as large as possible, i.e., s_large = 1/3 - epsilon.
    """
    n = 10
    tol = 1e-7
    
    # Set epsilon to ensure s_small > 0 and satisfy tolerance
    eps = 1e-5
    s_large = (1.0 / 3.0) - eps
    s_small = 1.0 - 3.0 * s_large
    
    if s_small <= 0:
        # Fallback
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides

    centers = np.zeros((n, 2), dtype=np.float64)
    sides = np.zeros(n, dtype=np.float64)
    
    # Shelf 1 (Bottom): 3 squares of size s_large
    y_center = s_large / 2.0
    x_centers = [s_large / 2.0, 3.0 * s_large / 2.0, 5.0 * s_large / 2.0]
    
    for i in range(3):
        centers[i, 0] = x_centers[i]
        centers[i, 1] = y_center
        sides[i] = s_large
        
    # Shelf 2: 3 squares of size s_large
    y_center = s_large + s_large / 2.0
    for i in range(3):
        idx = 3 + i
        centers[idx, 0] = x_centers[i]
        centers[idx, 1] = y_center
        sides[idx] = s_large
        
    # Shelf 3: 3 squares of size s_large
    y_center = 2.0 * s_large + s_large / 2.0
    for i in range(3):
        idx = 6 + i
        centers[idx, 0] = x_centers[i]
        centers[idx, 1] = y_center
        sides[idx] = s_large
        
    # Shelf 4 (Top): 1 square of size s_small
    y_center = 3.0 * s_large + s_small / 2.0
    centers[9, 0] = 0.5
    centers[9, 1] = y_center
    sides[9] = s_small
    
    return centers, sides

# EVOLVE-BLOCK-END
