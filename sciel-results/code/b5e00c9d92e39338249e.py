# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment


def szemeredi_trotter_incidences():
    """
    Construct 54 points and 27 lines to maximize point-line incidences.

    Strategy:
      - Start with a 9x9 grid of points (81 points).
      - Define 27 candidate lines: 9 vertical, 9 horizontal, 9 main diagonals.
      - Select 54 points from the 81 to maximize the total number of incidences
        with the 27 lines.
      - Use maximum weight bipartite matching / assignment to select the best subset.

    The 27 lines are:
      - 9 vertical lines: x = i for i in 1..9
      - 9 horizontal lines: y = j for j in 1..9
      - 9 diagonal lines: y - x = d for d in -8..8 (only those that pass through the grid)

    Each point (x, y) in the 9x9 grid lies on:
      - exactly 1 vertical line (x = x)
      - exactly 1 horizontal line (y = y)
      - exactly 1 diagonal line (y - x = y - x)

    So each point is incident to exactly 3 lines. If we select 54 points, the total
    incidences would be 54 * 3 = 162 if every selected point contributes 3 incidences.
    However, we need to verify that the lines are distinct and valid.

    Actually, let's reconsider. The 9 diagonals y - x = d for d in -8..8:
      - For d = -8: only point (9, 1)
      - For d = -7: points (8, 1), (9, 2)
      - ...
      - For d = 0: points (1,1), (2,2), ..., (9,9)
      - ...
      - For d = 8: only point (1, 9)

    All 17 diagonals (d from -8 to 8) are distinct. But we only want 9 diagonals.
    Let's pick the 9 diagonals with the most points: d = -4, -3, -2, -1, 0, 1, 2, 3, 4.
    These have 5, 6, 7, 8, 9, 8, 7, 6, 5 points respectively.

    Wait, the directive says "the 9 main diagonals of the full grid". Let's interpret
    "main diagonals" as the 9 diagonals with the largest number of points, which are
    d = -4, -3, -2, -1, 0, 1, 2, 3, 4.

    Now, each point (x, y) is on:
      - vertical line x = x
      - horizontal line y = y
      - diagonal line y - x = d where d = y - x, IF d is in {-4, -3, -2, -1, 0, 1, 2, 3, 4}

    So a point contributes 3 incidences if |y - x| <= 4, else 2 incidences.

    To maximize total incidences, we should select the 54 points with the highest
    individual incidence counts. Points with |y - x| <= 4 contribute 3, others contribute 2.

    Count of points with |y - x| <= 4 in a 9x9 grid:
      d = -4: 5 points
      d = -3: 6 points
      d = -2: 7 points
      d = -1: 8 points
      d = 0: 9 points
      d = 1: 8 points
      d = 2: 7 points
      d = 3: 6 points
      d = 4: 5 points
      Total = 5+6+7+8+9+8+7+6+5 = 61 points

    So we can select all 61 points with 3 incidences, but we only need 54. So we select
    54 of these 61 points, each contributing 3 incidences. Total = 54 * 3 = 162.

    But wait, we must also ensure that the 27 lines are all distinct and valid.
    The 9 vertical, 9 horizontal, and 9 diagonal lines are all distinct.

    Let's implement this:
      1. Generate all 81 points in the 9x9 grid.
      2. Compute the incidence count for each point with the 27 lines.
      3. Select the 54 points with the highest incidence counts.
      4. Return the selected points and the 27 lines.

    This should give 162 incidences, which is much better than the baseline 81.
    """
    # Generate 9x9 grid points
    xs = np.arange(1, 10)
    ys = np.arange(1, 10)
    PX, PY = np.meshgrid(xs, ys, indexing="ij")
    all_points = np.stack([PX.ravel(), PY.ravel()], axis=1).astype(np.int64)

    # Define 27 lines: 9 vertical, 9 horizontal, 9 diagonals
    lines = []
    # Vertical lines: x = i -> 1*x + 0*y - i = 0
    for i in range(1, 10):
        lines.append((1, 0, -i))
    # Horizontal lines: y = j -> 0*x + 1*y - j = 0
    for j in range(1, 10):
        lines.append((0, 1, -j))
    # Diagonal lines: y - x = d -> -1*x + 1*y - d = 0
    # Choose d in {-4, -3, -2, -1, 0, 1, 2, 3, 4}
    for d in range(-4, 5):
        lines.append((-1, 1, -d))

    lines = np.array(lines, dtype=np.int64)

    # Compute incidence count for each point
    # For each point, check which lines it lies on
    incidence_counts = np.zeros(len(all_points), dtype=np.int64)
    for idx, (x, y) in enumerate(all_points):
        for A, B, C in lines:
            if A * x + B * y + C == 0:
                incidence_counts[idx] += 1

    # Select the 54 points with the highest incidence counts
    # If there are ties, break them arbitrarily (e.g., by index)
    sorted_indices = np.argsort(-incidence_counts)
    selected_indices = sorted_indices[:54]
    points = all_points[selected_indices]

    return points, lines


# EVOLVE-BLOCK-END
