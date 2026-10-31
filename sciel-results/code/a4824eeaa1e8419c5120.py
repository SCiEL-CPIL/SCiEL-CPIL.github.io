"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time


def construct():
    """
    Pack 21 circles in a rectangle of perimeter 4 to maximize sum of radii.
    Uses a soft-body penalty approach with differential evolution.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Variables: for each circle, (x, y, r) plus rectangle width w (height determined by perimeter)
    # Perimeter = 2*(w + h) = 4 => h = 2 - w
    # So w in (0, 2), h = 2 - w
    # Total variables: 21*3 + 1 = 64

    def objective(params):
        """
        params: array of length 64
        First 63 values: for each circle i (0..20), x_i, y_i, r_i
        Last value: w (width of rectangle), h = 2 - w
        """
        if time.time() > deadline:
            return 1e10  # Large penalty to stop

        w = params[63]
        h = 2.0 - w

        # Ensure valid rectangle dimensions
        if w <= 0 or h <= 0:
            return 1e10

        # Extract circle parameters
        circles = params[:63].reshape(N, 3)
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]

        # Penalty for negative radii
        pen = 0.0
        for r in rs:
            if r < 0:
                pen += 100.0 * r * r

        # Penalty for circles going outside the rectangle
        for i in range(N):
            if xs[i] < rs[i]:
                pen += 100.0 * (rs[i] - xs[i]) ** 2
            if xs[i] > w - rs[i]:
                pen += 100.0 * (xs[i] - (w - rs[i])) ** 2
            if ys[i] < rs[i]:
                pen += 100.0 * (rs[i] - ys[i]) ** 2
            if ys[i] > h - rs[i]:
                pen += 100.0 * (ys[i] - (h - rs[i])) ** 2

        # Penalty for circle-circle overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    pen += 100.0 * (min_dist - dist) ** 2

        # Objective: maximize sum of radii, minimize penalty
        sum_r = np.sum(rs)
        return pen - sum_r

    # Bounds for differential evolution
    # For each circle: x in [0, 2], y in [0, 2], r in [0, 1]
    # w in [0.5, 1.5] (reasonable aspect ratios)
    bounds = []
    for i in range(N):
        bounds.append((0.0, 2.0))  # x
        bounds.append((0.0, 2.0))  # y
        bounds.append((0.0, 1.0))  # r
    bounds.append((0.5, 1.5))  # w

    # Run differential evolution with limited budget
    best_result = None
    best_score = -1e10

    try:
        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=100,
            popsize=15,
            tol=1e-4,
            atol=1e-4,
            seed=42,
            polish=True,
            init='sobol'
        )

        if time.time() < deadline - 5:
            best_result = result.x
            best_score = -result.fun
    except Exception:
        pass

    # If we haven't found a good result, try a simpler approach
    if best_result is None or best_score < 0:
        # Fallback: simple grid-based packing
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N] + [[1.0]])  # w=1, h=1
        best_score = sum(c[2] for c in cells[:N])

    # Extract the best result
    w = best_result[63]
    h = 2.0 - w
    circles = best_result[:63].reshape(N, 3)

    # Clamp circles to be inside the rectangle
    for i in range(N):
        circles[i, 0] = max(circles[i, 2], min(circles[i, 0], w - circles[i, 2]))
        circles[i, 1] = max(circles[i, 2], min(circles[i, 1], h - circles[i, 2]))
        circles[i, 2] = max(0.0, circles[i, 2])

    # Resolve any remaining overlaps by shrinking radii
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = circles[i, 2] + circles[j, 2]
            if dist < min_dist and dist > 0:
                # Shrink both radii to just fit
                shrink = (min_dist - dist) / 2 + 1e-6
                circles[i, 2] = max(0.0, circles[i, 2] - shrink)
                circles[j, 2] = max(0.0, circles[j, 2] - shrink)

    return circles

# EVOLVE-BLOCK-END
