# EVOLVE-BLOCK-START
import numpy as np


def szemeredi_trotter_incidences():
    """
    Constructs a set of 54 points and 27 lines to maximize incidences.
    Uses 9 vertical, 9 horizontal, and 9 diagonal lines.
    The diagonal lines are chosen to maximize the number of triple intersections
    within the 9x9 grid of vertical/horizontal intersections.
    This yields 61 triple-intersection points. We select 54 of them.
    Total incidences: 54 * 3 = 162.
    """
    lines = []
    
    # 9 vertical lines: x = 0..8  ->  (1, 0, -x)
    for i in range(9):
        lines.append((1, 0, -i))
        
    # 9 horizontal lines: y = 0..8  ->  (0, 1, -y)
    for j in range(9):
        lines.append((0, 1, -j))
        
    # 9 diagonal lines: y = x + k -> x - y + k = 0 -> (1, -1, k)
    # We choose k to maximize the number of (i,j) in [0,8]x[0,8] such that j-i = k
    # Counts for k in -4..4:
    # k=0: 9
    # k=1, -1: 8 each
    # k=2, -2: 7 each
    # k=3, -3: 6 each
    # k=4, -4: 5 each
    # Total 61 points.
    k_values = [0, 1, -1, 2, -2, 3, -3, 4, -4]
    for k in k_values:
        lines.append((1, -1, k))
        
    lines = np.array(lines, dtype=np.int64)
    
    points = []
    # Generate all 81 grid points, filter those on our diagonal lines
    for i in range(9):
        for j in range(9):
            k = j - i
            if k in k_values:
                points.append((i, j))
                
    # We have 61 points, need 54
    points = points[:54]
    points = np.array(points, dtype=np.int64)
    
    return points, lines


# EVOLVE-BLOCK-END
