# EVOLVE-BLOCK-START
import numpy as np


def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for 54 points and 27 lines.
    
    Uses a 7x8 grid construction (56 points) with 2 points removed.
    Points: {1,...,7} x {1,...,8}
    Lines selected to maximize density:
    - 7 vertical lines (x=1..7), each with 8 points.
    - 8 horizontal lines (y=1..8), each with 7 points.
    - 2 positive diagonals (y-x=c for c in {0,1}), each with 7 points.
    - 2 negative diagonals (y+x=c for c in {8,9}), each with 7 points.
    - 2 positive diagonals (y-x=c for c in {-1,2}), each with 6 points.
    - 2 negative diagonals (y+x=c for c in {7,10}), each with 6 points.
    - 2 positive diagonals (y-x=c for c in {-2,3}), each with 5 points.
    - 2 negative diagonals (y+x=c for c in {6,11}), each with 5 points.
    
    Total lines: 7 + 8 + 2 + 2 + 2 + 2 + 2 + 2 = 27.
    Total incidences on full 56-point grid:
    7*8 + 8*7 + 2*7 + 2*7 + 2*6 + 2*6 + 2*5 + 2*5 = 56 + 56 + 14 + 14 + 12 + 12 + 10 + 10 = 184.
    
    We remove 2 points with the minimum number of incidences to keep 54 points.
    """
    # Points: 7x8 grid
    xs = np.arange(1, 8)
    ys = np.arange(1, 9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    all_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

    lines = []

    # 7 Vertical lines: x = i -> 1*x + 0*y - i = 0
    for i in range(1, 8):
        lines.append((1, 0, -i))

    # 8 Horizontal lines: y = j -> 0*x + 1*y - j = 0
    for j in range(1, 9):
        lines.append((0, 1, -j))

    # 2 Positive diagonals (density 7): y - x = c
    for c in [0, 1]:
        lines.append((-1, 1, -c))

    # 2 Negative diagonals (density 7): y + x = c
    for c in [8, 9]:
        lines.append((1, 1, -c))

    # 2 Positive diagonals (density 6): y - x = c
    for c in [-1, 2]:
        lines.append((-1, 1, -c))

    # 2 Negative diagonals (density 6): y + x = c
    for c in [7, 10]:
        lines.append((1, 1, -c))

    # 2 Positive diagonals (density 5): y - x = c
    for c in [-2, 3]:
        lines.append((-1, 1, -c))

    # 2 Negative diagonals (density 5): y + x = c
    for c in [6, 11]:
        lines.append((1, 1, -c))

    lines = np.array(lines, dtype=np.int64)

    # Calculate incidence count for each point
    # A*x + B*y + C == 0
    # points: (56, 2), lines: (27, 3)
    # We can do this vectorized or loop. 56*27 is small.
    counts = np.zeros(len(all_points), dtype=np.int64)
    for p in all_points:
        x, y = p
        # Check all lines
        # vals = A*x + B*y + C
        vals = lines[:, 0] * x + lines[:, 1] * y + lines[:, 2]
        counts[np.where(p == all_points[0])[0][0]] = np.sum(vals == 0) # This is wrong, need to map back

    # Let's do it properly
    counts = np.zeros(len(all_points), dtype=np.int64)
    for i in range(len(all_points)):
        x, y = all_points[i]
        vals = lines[:, 0] * x + lines[:, 1] * y + lines[:, 2]
        counts[i] = np.sum(vals == 0)

    # Find 2 points with minimum counts
    # If tie, any is fine.
    # We want to remove points with smallest counts.
    idxs = np.argsort(counts)
    remove_idxs = idxs[:2]
    
    # Create mask to keep points
    mask = np.ones(len(all_points), dtype=bool)
    mask[remove_idxs] = False
    
    points = all_points[mask]
    
    return points, lines


# EVOLVE-BLOCK-END
