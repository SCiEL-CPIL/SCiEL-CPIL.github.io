# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares inside the unit square [0,1]^2 to maximize
    the sum of their side lengths, subject to non-overlapping interiors.
    Uses a multi-start local optimization approach with a tight feasibility check.
    """
    n = 10
    tol = 1e-7
    deadline = time.time() + 620  # leave 10s margin before 630s kill

    def feasibility_penalty(centers, sides):
        """Returns 0 if feasible, else a large penalty."""
        # Boundary check
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s / 2 < -tol or cx + s / 2 > 1 + tol:
                return 1e6
            if cy - s / 2 < -tol or cy + s / 2 > 1 + tol:
                return 1e6
        # Overlap check
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i][0] - centers[j][0]) < t and abs(centers[i][1] - centers[j][1]) < t:
                    return 1e6
        return 0.0

    def objective(x):
        """Minimize negative sum of sides + penalty."""
        centers = x[:2 * n].reshape(n, 2)
        sides = x[2 * n:]
        # Ensure sides are positive
        if np.any(sides <= 0):
            return 1e6
        pen = feasibility_penalty(centers, sides)
        return -np.sum(sides) + pen

    # Initial feasible grid solution
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx_init = (cols + 0.5) * side
    cy_init = (rows + 0.5) * side
    centers_init = np.stack([cx_init, cy_init], axis=1).astype(np.float64)
    sides_init = np.full(n, side, dtype=np.float64)

    def pack(centers, sides):
        x = np.concatenate([centers.flatten(), sides])
        return x

    x0 = pack(centers_init, sides_init)
    best_x = x0.copy()
    best_score = -np.sum(sides_init)

    # Bounds: centers in [0,1], sides in (0, 1]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # cx
        bounds.append((0.0, 1.0))  # cy
    for i in range(n):
        bounds.append((1e-9, 1.0))  # side

    # Multi-start: try several perturbations and initial configurations
    rng = np.random.default_rng(42)

    # Start 1: grid
    starts = [x0.copy()]

    # Start 2: random feasible-ish
    for trial in range(5):
        # Random sides summing to <= something, random centers
        sides_r = rng.uniform(0.1, 0.4, n)
        # Scale to fit roughly
        max_side = 0.35
        sides_r = np.clip(sides_r, 0.05, max_side)
        centers_r = np.zeros((n, 2))
        for i in range(n):
            s = sides_r[i]
            centers_r[i, 0] = rng.uniform(s / 2, 1 - s / 2)
            centers_r[i, 1] = rng.uniform(s / 2, 1 - s / 2)
        x_r = pack(centers_r, sides_r)
        starts.append(x_r)

    # Start 3: two rows of 5
    centers_2r = np.zeros((n, 2))
    sides_2r = np.full(n, 0.2)
    for i in range(5):
        centers_2r[i, 0] = 0.1 + i * 0.2
        centers_2r[i, 1] = 0.1
        centers_2r[i + 5, 0] = 0.1 + i * 0.2
        centers_2r[i + 5, 1] = 0.9
    starts.append(pack(centers_2r, sides_2r))

    # Start 4: one big + nine small
    centers_m = np.zeros((n, 2))
    sides_m = np.full(n, 0.1)
    sides_m[0] = 0.5
    centers_m[0, 0] = 0.5
    centers_m[0, 1] = 0.5
    # Place small ones around
    small_positions = [
        (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),
        (0.1, 0.5), (0.9, 0.5), (0.5, 0.1), (0.5, 0.9), (0.1, 0.1)
    ]
    for i in range(1, n):
        centers_m[i] = small_positions[i - 1]
    starts.append(pack(centers_m, sides_m))

    for x_start in starts:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, x_start, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.success or res.fun < best_score:
                x_new = res.x
                centers_new = x_new[:2 * n].reshape(n, 2)
                sides_new = x_new[2 * n:]
                if np.all(sides_new > 0) and feasibility_penalty(centers_new, sides_new) == 0:
                    score_new = -res.fun
                    if score_new > best_score:
                        best_score = score_new
                        best_x = x_new
        except Exception:
            pass

    # Final result
    centers = best_x[:2 * n].reshape(n, 2)
    sides = best_x[2 * n:]

    # Ensure validity
    if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(sides))):
        # Fallback to grid
        centers = centers_init
        sides = sides_init

    return centers, sides


# EVOLVE-BLOCK-END
