# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def squares_in_square10() -> tuple:
    n = 10
    deadline = time.time() + 580  # safety margin before 630s limit

    def _is_feasible(centers, sides, tol=1e-7):
        if centers.shape != (n, 2) or sides.shape != (n,):
            return False
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        if np.any(centers[:, 0] - sides / 2 < -tol) or np.any(centers[:, 0] + sides / 2 > 1 + tol):
            return False
        if np.any(centers[:, 1] - sides / 2 < -tol) or np.any(centers[:, 1] + sides / 2 > 1 + tol):
            return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                    return False
        return True

    def _to_x(centers, sides):
        x = np.empty(3 * n)
        x[:2 * n] = centers.flatten()
        x[2 * n:] = sides
        return x

    def _from_x(x):
        centers = x[:2 * n].reshape(n, 2)
        sides = x[2 * n:]
        return centers, sides

    # Analytical seed: 2 large (0.5) + 8 small (0.25) -> sum = 3.0
    def _seed_2_8():
        centers = np.zeros((n, 2), dtype=np.float64)
        sides = np.zeros(n, dtype=np.float64)
        centers[0] = [0.25, 0.25]; sides[0] = 0.5
        centers[1] = [0.75, 0.25]; sides[1] = 0.5
        xs = [0.125, 0.375, 0.625, 0.875]
        ys = [0.625, 0.875]
        idx = 2
        for y in ys:
            for x in xs:
                centers[idx] = [x, y]; sides[idx] = 0.25
                idx += 1
        return centers, sides

    def _random_seed(rng):
        # Try to build a random non-overlapping packing with random sizes
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        # Random sizes between 0.1 and 0.4
        s = rng.uniform(0.10, 0.35, n)
        # Sort by size descending to place large first
        order = np.argsort(-s)
        placed = False
        for _ in range(200):
            centers = np.zeros((n, 2))
            sides = np.zeros(n)
            ok = True
            for i in order:
                si = s[i]
                # Try random positions
                placed_i = False
                for _ in range(200):
                    cx = rng.uniform(si / 2, 1 - si / 2)
                    cy = rng.uniform(si / 2, 1 - si / 2)
                    centers[i] = [cx, cy]
                    sides[i] = si
                    # Check against already placed
                    conflict = False
                    for j in range(n):
                        if j == i or sides[j] == 0:
                            continue
                        sj = sides[j]
                        t = (si + sj) / 2
                        if abs(cx - centers[j, 0]) < t and abs(cy - centers[j, 1]) < t:
                            conflict = True
                            break
                    if not conflict:
                        placed_i = True
                        break
                if not placed_i:
                    ok = False
                    break
            if ok:
                placed = True
                break
        if not placed:
            # Fallback: grid
            k = 4
            side = 0.25
            idx = 0
            for r in range(3):
                for c in range(4):
                    if idx >= n:
                        break
                    centers[idx] = [(c + 0.5) * side, (r + 0.5) * side]
                    sides[idx] = side
                    idx += 1
            # Fill remaining with tiny squares in corners if needed
            while idx < n:
                centers[idx] = [0.02, 0.98]
                sides[idx] = 0.03
                idx += 1
        return centers, sides

    best_centers, best_sides = _seed_2_8()
    best_obj = float(np.sum(best_sides))

    # SLSQP constraints
    # Inequality constraints: g(x) >= 0
    def _make_constraints():
        constraints = []
        # Boundary: cx - s/2 >= 0, cx + s/2 <= 1 => 1 - cx - s/2 >= 0
        # cy - s/2 >= 0, cy + s/2 <= 1
        for i in range(n):
            # left
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * i] - x[2 * n + i] / 2})
            # right
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2 * i] - x[2 * n + i] / 2})
            # bottom
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * i + 1] - x[2 * n + i] / 2})
            # top
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2 * i + 1] - x[2 * n + i] / 2})
        # Pairwise: for each pair (i,j), we need |dx| >= t OR |dy| >= t
        # This is disjunctive. SLSQP can't handle OR directly.
        # Trick: use a smooth approximation or handle via penalty in objective.
        # Alternative: use the fact that for axis-aligned squares, non-overlap means:
        # cx_i - s_i/2 >= cx_j + s_j/2  OR  cx_j - s_j/2 >= cx_i + s_i/2  OR
        # cy_i - s_i/2 >= cy_j + s_j/2  OR  cy_j - s_j/2 >= cy_i + s_i/2
        # We'll use a penalty approach for overlaps in the objective instead of hard constraints.
        return constraints

    def _overlap_penalty(centers, sides, weight=1000.0):
        pen = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                # Violation: both dx < t and dy < t
                vx = max(0, t - dx)
                vy = max(0, t - dy)
                # If both positive, they overlap
                if vx > 0 and vy > 0:
                    pen += weight * (vx * vx + vy * vy)
        return pen

    def _objective(x):
        centers = x[:2 * n].reshape(n, 2)
        sides = x[2 * n:]
        if np.any(sides <= 0):
            return 1e10
        return -np.sum(sides) + _overlap_penalty(centers, sides, weight=1000.0)

    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # cx
        bounds.append((0.0, 1.0))  # cy
        bounds.append((0.01, 1.0))  # side

    # Phase 1: SLSQP from analytical seed
    x0 = _to_x(best_centers, best_sides)
    try:
        res = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                       options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False})
        if res.success or res.fun < 0:
            c_res, s_res = _from_x(res.x)
            if _is_feasible(c_res, s_res):
                obj_res = float(np.sum(s_res))
                if obj_res > best_obj:
                    best_centers = c_res.copy()
                    best_sides = s_res.copy()
                    best_obj = obj_res
    except Exception:
        pass

    # Phase 2: SLSQP from random seeds
    rng = np.random.default_rng(123)
    for attempt in range(20):
        if time.time() > deadline - 30:
            break
        c_rand, s_rand = _random_seed(rng)
        x_rand = _to_x(c_rand, s_rand)
        try:
            res = minimize(_objective, x_rand, method='SLSQP', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-10, 'disp': False})
            if res.success or res.fun < 0:
                c_res, s_res = _from_x(res.x)
                if _is_feasible(c_res, s_res):
                    obj_res = float(np.sum(s_res))
                    if obj_res > best_obj:
                        best_centers = c_res.copy()
                        best_sides = s_res.copy()
                        best_obj = obj_res
        except Exception:
            pass

    # Phase 3: Coordinate ascent polish on the best solution
    # For each square, try to increase its side while maintaining feasibility
    for _ in range(500):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            # Try increasing side
            for ds in [0.005, 0.002, 0.001, 0.0005]:
                s_new = best_sides[i] + ds
                if s_new > 1.0:
                    continue
                # Check if we can place square i with side s_new
                # The center must satisfy boundary and non-overlap with others
                # For now, just check if current center works
                c_new = best_centers.copy()
                s_new_arr = best_sides.copy()
                s_new_arr[i] = s_new
                # Clip center to boundary
                c_new[i, 0] = np.clip(c_new[i, 0], s_new / 2, 1 - s_new / 2)
                c_new[i, 1] = np.clip(c_new[i, 1], s_new / 2, 1 - s_new / 2)
                if _is_feasible(c_new, s_new_arr):
                    best_centers = c_new
                    best_sides = s_new_arr
                    best_obj = float(np.sum(best_sides))
                    improved = True
                    break
            # Try moving center to allow larger side
            for dx in [-0.002, 0.002]:
                for dy in [-0.002, 0.002]:
                    c_new = best_centers.copy()
                    s_new_arr = best_sides.copy()
                    s_new = s_new_arr[i] + 0.001
                    if s_new > 1.0:
                        continue
                    c_new[i, 0] = np.clip(c_new[i, 0] + dx, s_new / 2, 1 - s_new / 2)
                    c_new[i, 1] = np.clip(c_new[i, 1] + dy, s_new / 2, 1 - s_new / 2)
                    s_new_arr[i] = s_new
                    if _is_feasible(c_new, s_new_arr):
                        best_centers = c_new
                        best_sides = s_new_arr
                        best_obj = float(np.sum(best_sides))
                        improved = True
        if not improved:
            break

    # Final feasibility check
    if not _is_feasible(best_centers, best_sides):
        # Fallback to grid
        k = 4
        side = 0.25
        idx = 0
        for r in range(4):
            for c in range(4):
                if idx >= n:
                    break
                best_centers[idx] = [(c + 0.5) * side, (r + 0.5) * side]
                best_sides[idx] = side
                idx += 1

    return best_centers, best_sides


# EVOLVE-BLOCK-END
