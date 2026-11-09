# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def szemeredi_trotter_incidences():
    """
    Constructs a set of 54 points and 27 lines maximizing incidences.
    Uses a 6x9 grid with vertical, horizontal, and diagonal lines.
    """
    # Points: 6x9 grid
    # x in 1..6, y in 1..9
    xs = np.arange(1, 7)
    ys = np.arange(1, 10)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = []
    
    # 1. Vertical lines: x = i  ->  1*x + 0*y - i = 0
    for i in range(1, 7):
        lines.append((1, 0, -i))
        
    # 2. Horizontal lines: y = j  ->  0*x + 1*y - j = 0
    for j in range(1, 10):
        lines.append((0, 1, -j))
        
    # 3. Diagonal lines with slope 1 (y = x + c) -> 1*x - 1*y - c = 0
    # c in {0, 1, 2, 3} gives 6 points each
    for c in [0, 1, 2, 3]:
        lines.append((1, -1, -c))
        
    # 4. Diagonal lines with slope -1 (y = -x + c) -> 1*x + 1*y - c = 0
    # c in {7, 8, 9, 10} gives 6 points each
    for c in [7, 8, 9, 10]:
        lines.append((1, 1, -c))
        
    # 5. Additional diagonal lines with 5 points each
    # y = x - 1 (c = -1) -> 1*x - 1*y - (-1) = 0 -> (1, -1, 1)
    lines.append((1, -1, 1))
    
    # y = x + 4 (c = 4) -> 1*x - 1*y - 4 = 0 -> (1, -1, -4)
    lines.append((1, -1, -4))
    
    # y = -x + 6 (c = 6) -> 1*x + 1*y - 6 = 0 -> (1, 1, -6)
    lines.append((1, 1, -6))
    
    # y = -x + 11 (c = 11) -> 1*x + 1*y - 11 = 0 -> (1, 1, -11)
    lines.append((1, 1, -11))
    
    lines = np.array(lines, dtype=np.int64)
    
    return points, lines

# EVOLVE-BLOCK-END
