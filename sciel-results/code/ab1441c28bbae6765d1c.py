"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 21


def _validate(circles):
    """Check disjointness and perimeter constraint. Returns (is_valid, sum_radii)."""
    if circles is None or len(circles) != N:
        return False, 0.0
    if np.any(circles[:, 2] <= 0):
        return False, 0.0
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    # Bounding box
    x_min = np.min(xs - rs)
    x_max = np.max(xs + rs)
    y_min = np.min(ys - rs)
    y_max = np.max(ys + rs)
    width = x_max - x_min
    height = y_max - y_min
    perimeter = 2.0 * (width + height)
    if perimeter > 4.0 + 1e-9:
        return False, 0.0
    # Disjointness: for all pairs, dist >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-9:
                return False, 0.0
    return True, float(np.sum(rs))


def _force_directed_step(circles, width, height, lr):
    """
    One force-directed step. circles: (N,3) array [x, y, r].
    width, height: current rectangle dimensions (perimeter 2*(w+h)=4 => w+h=2).
    Returns updated circles.
    """
    n = len(circles)
    xs = circles[:, 0].copy()
    ys = circles[:, 1].copy()
    rs = circles[:, 2].copy()
    fx = np.zeros(n)
    fy = np.zeros(n)
    dr = np.zeros(n)

    # Pairwise repulsion
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-12:
                dx = 1e-6
                dy = 0.0
                dist = 1e-6
            overlap = rs[i] + rs[j] - dist
            if overlap > 0:
                # Push apart along direction
                nx = dx / dist
                ny = dy / dist
                # Move each by half the overlap, weighted by inverse radius
                wi = rs[j] / (rs[i] + rs[j])
                wj = rs[i] / (rs[i] + rs[j])
                fx[i] += overlap * nx * wi
                fy[i] += overlap * ny * wi
                fx[j] -= overlap * nx * wj
                fy[j] -= overlap * ny * wj
                # Also grow radii: reward for separation
                dr[i] += 0.01 * overlap
                dr[j] += 0.01 * overlap

    # Wall forces: push circles inward if they extend beyond [0, width] x [0, height]
    for i in range(n):
        # Left wall
        if xs[i] - rs[i] < 0:
            fx[i] += (0 - (xs[i] - rs[i]))
        # Right wall
        if xs[i] + rs[i] > width:
            fx[i] -= (xs[i] + rs[i] - width)
        # Bottom wall
        if ys[i] - rs[i] < 0:
            fy[i] += (0 - (ys[i] - rs[i]))
        # Top wall
        if ys[i] + rs[i] > height:
            fy[i] -= (ys[i] + rs[i] - height)

    # Update positions
    xs += lr * fx
    ys += lr * fy
    rs += lr * dr

    # Clamp radii to positive
    rs = np.maximum(rs, 0.001)

    return np.column_stack([xs, ys, rs])


def _grid_initialization(width, height, n):
    """Create a grid of n circles in a width x height rectangle."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            x = (i + 0.5) * width / side
            y = (j + 0.5) * height / side
            r = min(width, height) / side * 0.45
            cells.append([x, y, r])
    return np.array(cells[:n])


def _optimize(width, height, circles, time_budget):
    """
    Optimize circles in a width x height rectangle (w+h=2).
    Uses force-directed method with adaptive learning rate.
    Returns best circles found.
    """
    start_time = time.time()
    best_circles = circles.copy()
    best_score = np.sum(circles[:, 2])

    lr = 0.05
    for step in range(50000):
        if time.time() - start_time > time_budget:
            break

        circles = _force_directed_step(circles, width, height, lr)

        # Check validity
        valid, score = _validate(circles)
        if valid and score > best_score:
            best_score = score
            best_circles = circles.copy()

        # Decay learning rate
        lr *= 0.9999

        # Periodic restart with small perturbation if stuck
        if step % 1000 == 0 and step > 0:
            # Add small random perturbation
            circles = best_circles + 0.01 * np.random.randn(N, 3)
            circles[:, 2] = np.maximum(circles[:, 2], 0.001)

    return best_circles


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s

    # Try different aspect ratios (width + height = 2)
    best_overall = None
    best_score = 0.0

    # Aspect ratios to try: width from 0.5 to 1.5 (since height = 2 - width)
    aspect_ratios = [1.0, 0.8, 1.25, 0.6, 1.6, 0.4, 2.0]

    for width in aspect_ratios:
        if time.time() > deadline:
            break
        height = 2.0 - width
        if height <= 0.1:
            continue

        # Initialize with grid
        circles = _grid_initialization(width, height, N)

        # Optimize
        time_budget = min(15.0, deadline - time.time())
        if time_budget < 1.0:
            break
        result = _optimize(width, height, circles, time_budget)

        valid, score = _validate(result)
        if valid and score > best_score:
            best_score = score
            best_overall = result.copy()

    # If no valid result, fall back to simple grid
    if best_overall is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_overall = np.array(cells[:N])

    return best_overall


# EVOLVE-BLOCK-START
# (The entire implementation is above; this block marker is for compatibility)
# EVOLVE-BLOCK-END
