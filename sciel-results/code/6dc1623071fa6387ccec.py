# EVOLVE-BLOCK-START
import numpy as np


def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for n=54 points and m=27 lines.
    Uses a 9x6 grid construction with axis-aligned and diagonal lines.
    """
    # Points: 9x6 grid
    xs = np.arange(1, 10)  # 1 to 9
    ys = np.arange(1, 7)   # 1 to 6
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

    lines = []

    # 6 Horizontal lines: y = c -> 0*x + 1*y - c = 0
    for c in range(1, 7):
        lines.append((0, 1, -c))

    # 9 Vertical lines: x = c -> 1*x + 0*y - c = 0
    for c in range(1, 10):
        lines.append((1, 0, -c))

    # 4 Slope 1 lines (length 6): x - y - c = 0
    # c in {0, -1, -2, -3}
    for c in [0, -1, -2, -3]:
        lines.append((1, -1, -c))

    # 4 Slope -1 lines (length 6): x + y - c = 0
    # c in {7, 8, 9, 10}
    for c in [7, 8, 9, 10]:
        lines.append((1, 1, -c))

    # 4 Slope 1 lines (length 5): x - y - c = 0
    # c in {1, 2, 3, -4}
    for c in [1, 2, 3, -4]:
        lines.append((1, -1, -c))

    lines = np.array(lines, dtype=np.int64)

    return points, lines


# EVOLVE-BLOCK-END
