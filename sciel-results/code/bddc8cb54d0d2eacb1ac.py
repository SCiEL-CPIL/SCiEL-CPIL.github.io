# EVOLVE-BLOCK-START
import numpy as np


def szemeredi_trotter_incidences():
    """
    Construct a set of 54 points and 27 lines in the integer plane to maximize
    point-line incidences.
    
    Strategy: Use a 7x8 grid of points, removing 2 corner points to get 54 points.
    Points: x in {1..7}, y in {1..8}, excluding (1,1) and (7,8).
    Lines:
    1. 8 horizontal lines y = j (j=1..8).
    2. 7 vertical lines x = i (i=1..7).
    3. 6 diagonal lines with slope 1 (y = x + b).
       Selected b values to maximize incidences: 0, 1, -1, 2, -2, 3.
    4. 6 diagonal lines with slope -1 (y = -x + s, or x+y=s).
       Selected s values to maximize incidences: 9, 8, 10, 7, 11, 6.
       
    Total incidences:
    H: 54
    V: 54
    D1: 34
    D-1: 36
    Total: 178
    """
    # Points: 7x8 grid minus (1,1) and (7,8)
    xs = np.arange(1, 8)
    ys = np.arange(1, 9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Create mask to remove (1,1) and (7,8)
    mask = ~((points[:, 0] == 1) & (points[:, 1] == 1)) & ~((points[:, 0] == 7) & (points[:, 1] == 8))
    points = points[mask]
    
    lines = []
    
    # 8 Horizontal lines: y = j  =>  0x + 1y - j = 0
    for j in range(1, 9):
        lines.append((0, 1, -j))
        
    # 7 Vertical lines: x = i  =>  1x + 0y - i = 0
    for i in range(1, 8):
        lines.append((1, 0, -i))
        
    # 6 Diagonal lines slope 1: y = x + b  =>  1x - 1y + b = 0
    b_values = [0, 1, -1, 2, -2, 3]
    for b in b_values:
        lines.append((1, -1, b))
        
    # 6 Diagonal lines slope -1: x + y = s  =>  1x + 1y - s = 0
    s_values = [9, 8, 10, 7, 11, 6]
    for s in s_values:
        lines.append((1, 1, -s))
        
    lines = np.array(lines, dtype=np.int64)
    
    return points, lines


# EVOLVE-BLOCK-END
