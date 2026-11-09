# EVOLVE-BLOCK-START
import numpy as np

def szemeredi_trotter_incidences():
    """
    Constructs a 9x6 grid of points and a set of 27 lines maximizing incidences.
    
    Points: 9 columns (x=1..9) and 6 rows (y=1..6). Total 54 points.
    Lines:
    - 6 horizontal lines (y=const), each with 9 points.
    - 9 vertical lines (x=const), each with 6 points.
    - 4 diagonal lines (slope 1) of length 6.
    - 4 diagonal lines (slope -1) of length 6.
    - 2 diagonal lines (slope 1) of length 5.
    - 2 diagonal lines (slope -1) of length 5.
    
    Total incidences: 6*9 + 9*6 + 4*6 + 4*6 + 2*5 + 2*5 = 54 + 54 + 24 + 24 + 10 + 10 = 176.
    """
    # Points: 9x6 grid
    xs = np.arange(1, 10)  # 1 to 9
    ys = np.arange(1, 7)   # 1 to 6
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

    lines = []

    # 6 Horizontal lines: y = c -> 0x + 1y - c = 0
    for c in range(1, 7):
        lines.append((0, 1, -c))

    # 9 Vertical lines: x = c -> 1x + 0y - c = 0
    for c in range(1, 10):
        lines.append((1, 0, -c))

    # 4 Slope 1 lines (y - x = c) of length 6
    # c in {0, -1, -2, -3}
    for c in [0, -1, -2, -3]:
        lines.append((-1, 1, -c))

    # 4 Slope -1 lines (x + y = c) of length 6
    # c in {7, 8, 9, 10}
    for c in [7, 8, 9, 10]:
        lines.append((1, 1, -c))

    # 2 Slope 1 lines (y - x = c) of length 5
    # c in {1, -4}
    for c in [1, -4]:
        lines.append((-1, 1, -c))

    # 2 Slope -1 lines (x + y = c) of length 5
    # c in {6, 11}
    for c in [6, 11]:
        lines.append((1, 1, -c))

    lines = np.array(lines, dtype=np.int64)

    return points, lines

# EVOLVE-BLOCK-END
