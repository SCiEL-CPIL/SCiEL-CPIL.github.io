"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import random

_DEADLINE = 130.0  # seconds wall-clock safety margin


def _objective_and_feasibility(x, W):
    """
    x: array of shape (3N,) = [x0, y0, r0, x1, y1, r1, ...]
    W: rectangle width (height = 2 - W)
    Returns (objective_value, feasibility_score)
    objective_value = -sum(r_i)  (we minimize negative sum to maximize sum)
    feasibility_score = sum of constraint violations (0 if fully feasible)
    """
    H = 2.0 - W
    n = N
    coords = x.reshape(n, 3)
    xs = coords[:, 0]
    ys = coords[:, 1]
    rs = coords[:, 2]

    # Objective: maximize sum of radii -> minimize -sum(rs)
    obj = -np.sum(rs)

    # Constraints (violations):
    # 1) Each circle inside rectangle: xi >= ri, xi <= W-ri, yi >= ri, yi <= H-ri
    # 2) Pairwise non-overlap: dist_ij >= ri + rj

    # Boundary violations
    viol_left = np.maximum(0.0, rs - xs)
    viol_right = np.maximum(0.0, xs + rs - W)
    viol_bottom = np.maximum(0.0, rs - ys)
    viol_top = np.maximum(0.0, ys + rs - H)
    boundary_viol = np.sum(viol_left) + np.sum(viol_right) + np.sum(viol_bottom) + np.sum(viol_top)

    # Pairwise overlap violations
    overlap_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            req = rs[i] + rs[j]
            if dist < req:
                overlap_viol += (req - dist)

    feasibility = boundary_viol + overlap_viol
    return obj, feasibility


def _penalty(x, W, penalty_weight):
    """Penalized objective for scipy minimize."""
    obj, feas = _objective_and_feasibility(x, W)
    return obj + penalty_weight * feas


def _solve_for_width(W, max_iters=300, time_limit=10.0):
    """
    For a given rectangle width W (height = 2-W), try to find a good packing.
    Returns (best_sum_r, best_x) or (None, None).
    """
    H = 2.0 - W
    best_sum_r = 0.0
    best_x = None

    # Multiple random restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > _DEADLINE - 5.0:
            break

        # Initialize: place circles in a grid-like pattern with small radii
        side = int(np.ceil(np.sqrt(N)))
        rng = np.random.default_rng(seed=42 + restart * 1000 + int(W * 1000))
        x0 = np.zeros(3 * N)
        for i in range(N):
            row = i // side
            col = i % side
            cx = (col + 0.5) * W / side + rng.uniform(-0.1, 0.1) * W / side
            cy = (row + 0.5) * H / side + rng.uniform(-0.1, 0.1) * H / side
            r = 0.5 * min(W / side, H / side) * 0.8
            # Clamp
            cx = np.clip(cx, r, W - r)
            cy = np.clip(cy, r, H - r)
            x0[3 * i] = cx
            x0[3 * i + 1] = cy
            x0[3 * i + 2] = r

        # Gradually increase penalty weight
        penalty_weights = [10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]
        current_x = x0.copy()

        for pw in penalty_weights:
            if time.time() > _DEADLINE - 5.0:
                break
            res = minimize(
                _penalty,
                current_x,
                args=(W, pw),
                method='L-BFGS-B',
                options={'maxiter': max_iters // len(penalty_weights), 'ftol': 1e-12, 'gtol': 1e-10},
            )
            current_x = res.x
            # Project back to feasible region approximately
            obj, feas = _objective_and_feasibility(current_x, W)
            if feas < 1e-6:
                sum_r = -obj
                if sum_r > best_sum_r:
                    best_sum_r = sum_r
                    best_x = current_x.copy()

    return best_sum_r, best_x


def _refine(x, W, max_iters=500, time_limit=15.0):
    """
    Refine a feasible solution by maximizing sum of radii with constraints.
    Uses penalty method with increasing weights.
    """
    H = 2.0 - W
    current_x = x.copy()
    best_sum_r = -_objective_and_feasibility(current_x, W)[0]
    best_x = current_x.copy()

    penalty_weights = [100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0, 100000.0]
    for pw in penalty_weights:
        if time.time() > _DEADLINE - 5.0:
            break
        res = minimize(
            _penalty,
            current_x,
            args=(W, pw),
            method='L-BFGS-B',
            options={'maxiter': max_iters // len(penalty_weights), 'ftol': 1e-14, 'gtol': 1e-12},
        )
        current_x = res.x
        obj, feas = _objective_and_feasibility(current_x, W)
        if feas < 1e-5:
            sum_r = -obj
            if sum_r > best_sum_r:
                best_sum_r = sum_r
                best_x = current_x.copy()

    return best_sum_r, best_x


def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)

    best_sum_r = 0.0
    best_x = None
    best_W = 1.0

    # Search over different aspect ratios (widths)
    # Perimeter = 2*(W+H) = 4 => H = 2 - W, W in (0, 2)
    # Try a range of widths
    widths = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]

    for W in widths:
        if time.time() > _DEADLINE - 20.0:
            break

        sum_r, x_sol = _solve_for_width(W)
        if x_sol is not None and sum_r > best_sum_r:
            best_sum_r = sum_r
            best_x = x_sol
            best_W = W

        # Refine the best solution found for this width
        if x_sol is not None:
            sum_r2, x_refined = _refine(x_sol, W)
            if sum_r2 > best_sum_r:
                best_sum_r = sum_r2
                best_x = x_refined
                best_W = W

    # If we still have time, do additional random restarts on the best width
    if best_x is not None and time.time() < _DEADLINE - 30.0:
        for extra_restart in range(10):
            if time.time() > _DEADLINE - 10.0:
                break
            H = 2.0 - best_W
            rng = np.random.default_rng(seed=10000 + extra_restart)
            side = int(np.ceil(np.sqrt(N)))
            x0 = np.zeros(3 * N)
            for i in range(N):
                row = i // side
                col = i % side
                cx = (col + 0.5) * best_W / side + rng.uniform(-0.2, 0.2) * best_W / side
                cy = (row + 0.5) * H / side + rng.uniform(-0.2, 0.2) * H / side
                r = 0.5 * min(best_W / side, H / side) * 0.7
                cx = np.clip(cx, r, best_W - r)
                cy = np.clip(cy, r, H - r)
                x0[3 * i] = cx
                x0[3 * i + 1] = cy
                x0[3 * i + 2] = r

            sum_r, x_sol = _solve_for_width(best_W)
            # Use this new init directly
            penalty_weights = [100.0, 500.0, 1000.0, 5000.0, 10000.0]
            current_x = x0.copy()
            for pw in penalty_weights:
                if time.time() > _DEADLINE - 10.0:
                    break
                res = minimize(
                    _penalty,
                    current_x,
                    args=(best_W, pw),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12},
                )
                current_x = res.x
                obj, feas = _objective_and_feasibility(current_x, best_W)
                if feas < 1e-5:
                    sum_r_curr = -obj
                    if sum_r_curr > best_sum_r:
                        best_sum_r = sum_r_curr
                        best_x = current_x.copy()

    # Final safety: if we have a valid solution, return it; else return grid
    if best_x is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        W = 1.0
        H = 1.0
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                r = 0.5 / side - 1e-3
                cx = (i + 0.5) / side * W
                cy = (j + 0.5) / side * H
                cells.append([cx, cy, r])
        return np.array(cells[:N])

    # Validate: ensure perimeter constraint and disjoint interiors
    H = 2.0 - best_W
    coords = best_x.reshape(N, 3)
    # Clamp to be safe
    for i in range(N):
        coords[i, 0] = np.clip(coords[i, 0], coords[i, 2], best_W - coords[i, 2])
        coords[i, 1] = np.clip(coords[i, 1], coords[i, 2], H - coords[i, 2])

    # Check feasibility
    _, feas = _objective_and_feasibility(coords.ravel(), best_W)
    if feas > 1e-3:
        # If not feasible, fall back to grid
        side = int(np.ceil(np.sqrt(N)))
        W = 1.0
        H = 1.0
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                r = 0.5 / side - 1e-3
                cx = (i + 0.5) / side * W
                cy = (j + 0.5) / side * H
                cells.append([cx, cy, r])
        return np.array(cells[:N])

    return coords
# EVOLVE-BLOCK-END
