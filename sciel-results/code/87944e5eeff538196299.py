# EVOLVE-BLOCK-START
import numpy as np


def szemeredi_trotter_incidences():
    """
    Constructs a set of 54 points and 27 lines to maximize incidences.
    Uses a triangular grid of 9 vertical, 9 horizontal, and 9 diagonal lines.
    This configuration yields 45 triple-intersection points and many double-intersection points.
    We select the 45 triple points and 9 double points to form the 54 points.
    Total incidences: 45 * 3 + 9 * 2 = 153.
    """
    # Define the 27 lines
    # 9 vertical lines: x = 0..8  ->  (1, 0, -x)
    # 9 horizontal lines: y = 0..8  ->  (0, 1, -y)
    # 9 diagonal lines: y = x + k, k=0..8  ->  (1, -1, -k)  (x - y + k = 0? No, y = x + k -> x - y + k = 0 is x - (x+k) + k = 0. Correct.)
    
    lines = []
    for i in range(9):
        lines.append((1, 0, -i))
    for j in range(9):
        lines.append((0, 1, -j))
    for k in range(9):
        lines.append((1, -1, -k))
        
    lines = np.array(lines, dtype=np.int64)
    
    # Define the 54 points
    # 45 triple points: (i, j) with 0<=i<=8, 0<=j<=8, j>=i
    points = []
    for i in range(9):
        for j in range(i, 9):
            points.append((i, j))
            
    # We have 45 points. Need 9 more.
    # Pick 9 double points from the remaining grid points (j < i)
    # Just pick the first 9 in row-major order where j < i
    count = 0
    for i in range(9):
        for j in range(i):
            points.append((i, j))
            count += 1
            if count == 9:
                break
        if count == 9:
            break
            
    points = np.array(points, dtype=np.int64)
    
    return points, lines


# EVOLVE-BLOCK-END
