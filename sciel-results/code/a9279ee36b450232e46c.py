# EVOLVE-BLOCK-START
import numpy as np


def szemeredi_trotter_incidences():
    """
    Constructs a set of 54 points and 27 lines to maximize incidences.
    Uses 9 vertical, 9 horizontal, 6 diagonal (slope 1), and 3 diagonal (slope -1) lines.
    This mix creates 9 points on 4 lines, 42 points on 3 lines, and 3 points on 2 lines.
    Total incidences: 9 * 4 + 42 * 3 + 3 * 2 = 36 + 126 + 6 = 168.
    """
    lines = []
    
    # 9 vertical lines: x = 0..8  ->  (1, 0, -x)
    for i in range(9):
        lines.append((1, 0, -i))
        
    # 9 horizontal lines: y = 0..8  ->  (0, 1, -y)
    for j in range(9):
        lines.append((0, 1, -j))
        
    # 6 diagonal lines with slope 1: y = x + k -> x - y + k = 0 -> (1, -1, k)
    # k values chosen to maximize intersections with slope -1 lines
    k_values = [0, 1, -1, 2, -2, 3]
    for k in k_values:
        lines.append((1, -1, k))
        
    # 3 diagonal lines with slope -1: y = -x + s -> x + y - s = 0 -> (1, 1, -s)
    s_values = [7, 8, 9]
    for s in s_values:
        lines.append((1, 1, -s))
        
    lines = np.array(lines, dtype=np.int64)
    
    points = []
    # Generate all 81 grid points, calculate incidence count for each
    point_counts = []
    for i in range(9):
        for j in range(9):
            count = 2  # Always on one vertical and one horizontal
            k = j - i
            if k in k_values:
                count += 1
            s = i + j
            if s in s_values:
                count += 1
            point_counts.append((count, i, j))
    
    # Sort by count descending, then by i, j for determinism
    point_counts.sort(key=lambda x: (-x[0], x[1], x[2]))
    
    # Select top 54 points
    selected_points = [(i, j) for (_, i, j) in point_counts[:54]]
    points = np.array(selected_points, dtype=np.int64)
    
    return points, lines


# EVOLVE-BLOCK-END
