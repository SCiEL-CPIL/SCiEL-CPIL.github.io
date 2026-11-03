"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
EPS = 1e-7


# EVOLVE-BLOCK-START
def _perimeter(w, h):
    return 2.0 * (w + h)


def _check_validity(cx, cy, r, w, h):
    """Check if circles are valid: disjoint interiors and within rectangle."""
    if w <= 0 or h <= 0:
        return False
    if _perimeter(w, h) > 4.0 + 1e-6:
        return False
    # Check boundaries
    for i in range(len(r)):
        if cx[i] < r[i] - 1e-5 or cx[i] > w - r[i] + 1e-5:
            return False
        if cy[i] < r[i] - 1e-5 or cy[i] > h - r[i] + 1e-5:
            return False
    # Check pairwise non-overlap
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-5:
                return False
    return True


def _objective(x, w, h):
    """Objective to minimize: negative sum of radii + penalties for violations."""
    cx = x[:N]
    cy = x[N:2 * N]
    r = x[2 * N:]

    # Penalty for perimeter constraint
    perim = 2.0 * (w + h)
    perim_pen = 0.0
    if perim > 4.0:
        perim_pen = 1e6 * (perim - 4.0) ** 2

    # Penalty for boundary violations
    bound_pen = 0.0
    for i in range(N):
        if cx[i] < r[i]:
            bound_pen += 1e6 * (r[i] - cx[i]) ** 2
        if cx[i] > w - r[i]:
            bound_pen += 1e6 * (cx[i] - (w - r[i])) ** 2
        if cy[i] < r[i]:
            bound_pen += 1e6 * (r[i] - cy[i]) ** 2
        if cy[i] > h - r[i]:
            bound_pen += 1e6 * (cy[i] - (h - r[i])) ** 2

    # Penalty for overlap
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = np.sqrt(dx * dx + dy * dy)
            sep = r[i] + r[j]
            if dist < sep:
                overlap_pen += 1e6 * (sep - dist) ** 2

    obj = -np.sum(r) + perim_pen + bound_pen + overlap_pen
    return obj


def _local_optimize(cx, cy, r, w, h, deadline):
    """Local optimization using scipy minimize."""
    x0 = np.concatenate([cx, cy, r])

    def obj(x):
        return _objective(x, w, h)

    result = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10})
    x_opt = result.x
    cx_opt = x_opt[:N]
    cy_opt = x_opt[N:2 * N]
    r_opt = x_opt[2 * N:]
    return cx_opt, cy_opt, r_opt


def _make_initial_grid(n, w, h):
    """Create initial grid layout."""
    side = int(np.ceil(np.sqrt(n)))
    cx = np.zeros(n)
    cy = np.zeros(n)
    r = np.zeros(n)
    for i in range(n):
        row = i // side
        col = i % side
        cx[i] = (col + 0.5) * w / side
        cy[i] = (row + 0.5) * h / side
        r[i] = 0.5 * min(w, h) / side - 1e-3
    return cx, cy, r


def _make_staggered(n, w, h):
    """Create staggered hexagonal-like layout."""
    cx = np.zeros(n)
    cy = np.zeros(n)
    r = np.zeros(n)
    rows = int(np.ceil(np.sqrt(n * h / w)))
    cols = int(np.ceil(n / rows))
    dx = w / cols
    dy = h / rows
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            offset = dx / 2 if row % 2 == 1 else 0
            cx[idx] = (col + 0.5) * dx + offset
            if cx[idx] > w - 0.1:
                cx[idx] = w - 0.1
            cy[idx] = (row + 0.5) * dy
            r[idx] = 0.45 * min(dx, dy)
            idx += 1
    return cx, cy, r


def _evaluate_solution(cx, cy, r, w, h):
    """Evaluate and validate a solution, returning (value, valid)."""
    if not _check_validity(cx, cy, r, w, h):
        return 0.0, False
    return np.sum(r), True


def construct():
    deadline = time.time() + 110.0  # Safety margin before 120s limit

    best_value = 0.0
    best_result = None

    # Try different aspect ratios
    aspect_ratios = [1.0, 1.2, 1.5, 2.0, 0.8, 1.3, 1.8, 0.7, 1.1, 1.4]

    for aspect in aspect_ratios:
        if time.time() > deadline:
            break

        # w + h = 2 (perimeter = 4)
        # w/h = aspect => w = aspect*h => aspect*h + h = 2 => h = 2/(aspect+1)
        h = 2.0 / (aspect + 1.0)
        w = 2.0 - h

        # Try multiple initial layouts
        layouts = []

        # Grid layout
        cx_g, cy_g, r_g = _make_initial_grid(N, w, h)
        layouts.append((cx_g, cy_g, r_g, w, h))

        # Staggered layout
        cx_s, cy_s, r_s = _make_staggered(N, w, h)
        layouts.append((cx_s, cy_s, r_s, w, h))

        # Random layouts
        np.random.seed(42)
        for seed in range(3):
            if time.time() > deadline:
                break
            np.random.seed(seed * 1000 + int(aspect * 100))
            cx_r = np.random.uniform(0.1, w - 0.1, N)
            cy_r = np.random.uniform(0.1, h - 0.1, N)
            r_r = np.random.uniform(0.05, 0.15, N)
            layouts.append((cx_r, cy_r, r_r, w, h))

        for cx_init, cy_init, r_init, w_curr, h_curr in layouts:
            if time.time() > deadline:
                break

            # Local optimization
            cx_opt, cy_opt, r_opt = _local_optimize(cx_init, cy_init, r_init, w_curr, h_curr, deadline)

            # Ensure radii are positive
            r_opt = np.maximum(r_opt, 1e-6)

            value, valid = _evaluate_solution(cx_opt, cy_opt, r_opt, w_curr, h_curr)
            if valid and value > best_value:
                best_value = value
                best_result = np.column_stack([cx_opt, cy_opt, r_opt])

            # Also try the initial layout
            value_init, valid_init = _evaluate_solution(cx_init, cy_init, r_init, w_curr, h_curr)
            if valid_init and value_init > best_value:
                best_value = value_init
                best_result = np.column_stack([cx_init, cy_init, r_init])

    # If no valid solution found, use a simple fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
        # Scale to fit perimeter 4: square with side 1
        # Already in [0,1]x[0,1] which has perimeter 4

    return best_result
# EVOLVE-BLOCK-END
