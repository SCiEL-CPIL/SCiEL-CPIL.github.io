# EVOLVE-BLOCK-START
import numpy as np
import time

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines.
    Uses a 6x9 grid construction.
    Points: 6x9 grid (54 points).
    Lines: 
      6 vertical lines (x=1..6), each hits 9 points.
      9 horizontal lines (y=1..9), each hits 6 points.
      12 diagonal lines (slope 1 and -1) chosen to maximize incidences.
    Total incidences: 6*9 + 9*6 + (4*6 + 4*6 + 2*5 + 2*5) = 54 + 54 + 68 = 176.
    """
    # Points: 6x9 grid
    xs = np.arange(1, 7)
    ys = np.arange(1, 10)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

    lines = []

    # 6 Vertical lines: x = c => 1*x + 0*y - c = 0
    for c in range(1, 7):
        lines.append((1, 0, -c))

    # 9 Horizontal lines: y = c => 0*x + 1*y - c = 0
    for c in range(1, 10):
        lines.append((0, 1, -c))

    # 4 Slope 1 lines (max length 6): y = x + c => x - y + c = 0
    # c in {0, 1, 2, 3}
    for c in [0, 1, 2, 3]:
        lines.append((1, -1, c))

    # 4 Slope -1 lines (max length 6): y = -x + c => x + y - c = 0
    # c in {7, 8, 9, 10}
    for c in [7, 8, 9, 10]:
        lines.append((1, 1, -c))

    # 2 Slope 1 lines (length 5): c in {4, -1}
    for c in [4, -1]:
        lines.append((1, -1, c))

    # 2 Slope -1 lines (length 5): c in {6, 11}
    for c in [6, 11]:
        lines.append((1, 1, -c))

    lines = np.array(lines, dtype=np.int64)
    
    return points, lines

# EVOLVE-BLOCK-END
