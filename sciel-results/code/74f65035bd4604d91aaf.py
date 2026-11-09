# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    """
    n = 10
    tol = 1e-7
    deadline = time.time() + 565  # safety margin before 600s limit

    # ---------- feasibility ----------
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

    def _sum_sides(sides):
        return float(np.sum(sides))

    def _objective(centers, sides):
        if _is_feasible(centers, sides):
            return _sum_sides(sides)
        return -1e6

    # ---------- conversion helpers ----------
    def _to_x(centers, sides):
        x = np.empty(3 * n)
        x[:2 * n] = centers.flatten()
        x[2 * n:] = sides
        return x

    def _from_x(x):
        centers = x[:2 * n].reshape(n, 2)
        sides = x[2 * n:]
        return centers, sides

    # ---------- smooth (softplus) penalty for L-BFGS-B ----------
    def _softplus(x):
        return np.where(x > 30, x, np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0))

    def _smooth_penalty(x):
        centers = x[:2 * n].reshape(n, 2)
        sides = x[2 * n:]
        pen = 0.0
        pen += float(np.sum(_softplus(-(centers[:, 0] - sides / 2))))
        pen += float(np.sum(_softplus(centers[:, 0] + sides / 2 - 1)))
        pen += float(np.sum(_softplus(-(centers[:, 1] - sides / 2))))
        pen += float(np.sum(_softplus(centers[:, 1] + sides / 2 - 1)))
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                vx = t - dx
                vy = t - dy
                pen += _softplus(vx) + _softplus(vy) - _softplus(vx + vy)
        pen += float(np.sum(_softplus(-sides)))
        return pen

    def _scipy_objective(x):
        centers = x[:2 * n].reshape(n, 2)
        sides = x[2 * n:]
        if np.any(sides <= 0.001):
            return 1e6
        return -np.sum(sides) + 100.0 * _smooth_penalty(x)

    # ---------- repair: project to feasibility ----------
    def _repair(centers, sides, iters=200):
        centers = np.array(centers, dtype=np.float64)
        sides = np.array(sides, dtype=np.float64)
        for _ in range(iters):
            changed = False
            # clamp to walls
            for i in range(n):
                s = sides[i]
                lo_x, hi_x = s / 2, 1 - s / 2
                if centers[i, 0] < lo_x:
                    centers[i, 0] = lo_x; changed = True
                elif centers[i, 0] > hi_x:
                    centers[i, 0] = hi_x; changed = True
                lo_y, hi_y = s / 2, 1 - s / 2
                if centers[i, 1] < lo_y:
                    centers[i, 1] = lo_y; changed = True
                elif centers[i, 1] > hi_y:
                    centers[i, 1] = hi_y; changed = True
            # resolve overlaps by pushing apart along smaller axis
            for i in range(n):
                for j in range(i + 1, n):
                    t = (sides[i] + sides[j]) / 2
                    dx = centers[j, 0] - centers[i, 0]
                    dy = centers[j, 1] - centers[i, 1]
                    if abs(dx) < t and abs(dy) < t:
                        if abs(dx) <= abs(dy):
                            gap = t - abs(dx)
                            if dx >= 0:
                                centers[i, 0] -= gap / 2
                                centers[j, 0] += gap / 2
                            else:
                                centers[i, 0] += gap / 2
                                centers[j, 0] -= gap / 2
                        else:
                            gap = t - abs(dy)
                            if dy >= 0:
                                centers[i, 1] -= gap / 2
                                centers[j, 1] += gap / 2
                            else:
                                centers[i, 1] += gap / 2
                                centers[j, 1] -= gap / 2
                        changed = True
            # re-clamp after push
            for i in range(n):
                s = sides[i]
                centers[i, 0] = np.clip(centers[i, 0], s / 2, 1 - s / 2)
                centers[i, 1] = np.clip(centers[i, 1], s / 2, 1 - s / 2)
            if not changed:
                break
        return centers, sides

    # ---------- coordinate-ascent interval-arithmetic optimizer ----------
    def _ca_optimize(centers, sides, max_sweeps=400, eps=1e-8):
        centers = np.array(centers, dtype=np.float64)
        sides = np.array(sides, dtype=np.float64)
        if not _is_feasible(centers, sides):
            centers, sides = _repair(centers, sides)
        best_val = _sum_sides(sides)
        for _ in range(max_sweeps):
            if time.time() > deadline - 20:
                break
            improved = False
            for i in range(n):
                # --- maximize side from bottleneck gap ---
                old_s = sides[i]
                Lx = centers[i, 0] - old_s / 2
                Rx = centers[i, 0] + old_s / 2
                Ly = centers[i, 1] - old_s / 2
                Ry = centers[i, 1] + old_s / 2
                maxLx, maxRx = 0.0, 1.0
                maxLy, maxRy = 0.0, 1.0
                for j in range(n):
                    if j == i:
                        continue
                    Lj = centers[j, 0] - sides[j] / 2
                    Rj = centers[j, 0] + sides[j] / 2
                    Ljy = centers[j, 1] - sides[j] / 2
                    Rjy = centers[j, 1] + sides[j] / 2
                    if Ly < Rjy and Ry > Ljy:  # y-overlap -> x constraint
                        if Rj < centers[i, 0]:
                            maxLx = max(maxLx, Rj + eps)
                        else:
                            maxRx = min(maxRx, Lj - eps)
                    if Lx < Rj and Rx > Lj:  # x-overlap -> y constraint
                        if Rjy < centers[i, 1]:
                            maxLy = max(maxLy, Rjy + eps)
                        else:
                            maxRy = min(maxRy, Ljy - eps)
                max_sx = maxRx - maxLx
                max_sy = maxRy - maxLy
                new_s = min(max_sx, max_sy, 1.0)
                if new_s > old_s + 1e-9 and new_s > 1e-6:
                    sides[i] = new_s
                    improved = True
                # --- recenter to midpoint of feasible interval ---
                cx = (maxLx + maxRx) / 2
                cy = (maxLy + maxRy) / 2
                s = sides[i]
                cx = np.clip(cx, s / 2, 1 - s / 2)
                cy = np.clip(cy, s / 2, 1 - s / 2)
                if abs(cx - centers[i, 0]) > 1e-10 or abs(cy - centers[i, 1]) > 1e-10:
                    centers[i, 0] = cx
                    centers[i, 1] = cy
                    improved = True
            # safety repair if numerical drift caused infeasibility
            if not _is_feasible(centers, sides):
                centers, sides = _repair(centers, sides)
                if not _is_feasible(centers, sides):
                    break
            val = _sum_sides(sides)
            if val > best_val:
                best_val = val
            if not improved:
                break
        if _is_feasible(centers, sides):
            return centers, sides
        return None

    # ---------- diverse initial topologies ----------
    def _grid(k):
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        c = np.stack([cx, cy], axis=1)
        s = np.full(n, side)
        return c, s

    def _three_by_three_center():
        c = np.zeros((n, 2)); s = np.zeros(n)
        pos = [(0.5, 0.5)]
        side = 1.0 / 3
        for r in range(3):
            for col in range(3):
                if r == 1 and col == 1:
                    continue
                pos.append(((col + 0.5) * side, (r + 0.5) * side))
                s.append(side)
        s[0] = 1.0 / 3
        c = np.array(pos); s = np.array(s)
        return c, s

    def _two_by_five():
        c = np.zeros((n, 2)); s = np.zeros(n)
        side = 1.0 / 5
        for r in range(2):
            for col in range(5):
                c[len(s)] = ((col + 0.5) * side, (r + 0.5) * (1.0 / 2))
                s[len(s)] = side
        return c, s

    def _four_by_three():
        c = np.zeros((n, 2)); s = np.zeros(n)
        side = 1.0 / 4
        for r in range(3):
            for col in range(4):
                if len(s) >= n:
                    break
                c[len(s)] = ((col + 0.5) * side, (r + 0.5) * (1.0 / 3))
                s[len(s)] = side
            if len(s) >= n:
                break
        # fill any leftover with a small square in a corner
        while len(s) < n:
            c[len(s)] = (0.1, 0.1); s[len(s)] = 0.1
        return c[:n], s[:n]

    # ---------- main search ----------
    rng = np.random.default_rng(42)

    starts = []
    for gen in (_grid, _three_by_three_center, _two_by_five, _four_by_three):
        try:
            c0, s0 = gen()
            c0, s0 = _repair(c0, s0)
            if _is_feasible(c0, s0):
                starts.append((c0.copy(), s0.copy()))
        except Exception:
            pass
    # a few randomized starts
    for _ in range(4):
        c0 = rng.uniform(0.15, 0.85, (n, 2))
        s0 = rng.uniform(0.10, 0.28, n)
        c0, s0 = _repair(c0, s0)
        if _is_feasible(c0, s0):
            starts.append((c0.copy(), s0.copy()))

    best_centers = None
    best_sides = None
    best_obj = -1e6

    def _consider(c, s):
        nonlocal best_centers, best_sides, best_obj
        if c is None:
            return
        if not _is_feasible(c, s):
            return
        val = _sum_sides(s)
        if val > best_obj + 1e-12:
            best_obj = val
            best_centers = c.copy()
            best_sides = s.copy()

    for c0, s0 in starts:
        if time.time() > deadline - 40:
            break
        # coordinate-ascent interval optimizer (the strong move)
        ca_c, ca_s = _ca_optimize(c0, s0, max_sweeps=300)
        _consider(ca_c, ca_s)
        _consider(c0, s0)

        # random local search to escape local optima
        cur_c, cur_s = (best_centers.copy(), best_sides.copy()) if best_centers is not None else (c0.copy(), s0.copy())
        cur_c, cur_s = c0.copy(), s0.copy()
        step = 0.04
        for it in range(1500):
            if time.time() > deadline - 40:
                break
            xc = cur_c + rng.normal(0, step, (n, 2))
            xs = np.maximum(cur_s + rng.normal(0, step, n), 0.02)
            xc, xs = _repair(xc, xs)
            if _is_feasible(xc, xs):
                val = _sum_sides(xs)
                if val > best_obj + 1e-12:
                    _consider(xc, xs)
                    cur_c, cur_s = xc.copy(), xs.copy()
                    step *= 0.95
                elif rng.random() < 0.15:
                    cur_c, cur_s = xc.copy(), xs.copy()
                    step *= 1.05
            if it % 100 == 0:
                cur_c, cur_s = best_centers.copy(), best_sides.copy()

    if best_centers is None:
        c0, s0 = _repair(_grid(4)[0], _grid(4)[1])
        best_centers, best_sides = c0, s0
        best_obj = _sum_sides(s0)

    # L-BFGS-B polish from the best (and a couple of alternates) with smooth penalty
    if time.time() < deadline - 60:
        bounds = []
        for _ in range(n):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
            bounds.append((0.02, 1.0))
        start_list = [_to_x(best_centers, best_sides)]
        for c0, s0 in starts[:2]:
            start_list.append(_to_x(c0, s0))
        for x0 in start_list:
            if time.time() > deadline - 45:
                break
            try:
                res = minimize(_scipy_objective, x0, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-12,
                                        'maxls': 50, 'maxfun': 4000})
                c_res, s_res = _from_x(res.x)
                c_res, s_res = _repair(c_res, s_res)
                if _is_feasible(c_res, s_res):
                    val = _sum_sides(s_res)
                    if val > best_obj + 1e-12:
                        _consider(c_res, s_res)
                        # coordinate-ascent polish on the scipy result
                        ca_c, ca_s = _ca_optimize(c_res, s_res, max_sweeps=200)
                        _consider(ca_c, ca_s)
            except Exception:
                pass

    # final coordinate-ascent polish from the global best
    if time.time() < deadline - 10:
        ca_c, ca_s = _ca_optimize(best_centers, best_sides, max_sweeps=500)
        _consider(ca_c, ca_s)

    # final fine coordinate descent (tiny nudges)
    for _ in range(400):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(n):
            for dx in (-0.0003, 0.0003):
                for dy in (-0.0003, 0.0003):
                    for ds in (-0.0003, 0.0003):
                        c_new = best_centers.copy()
                        s_new = best_sides.copy()
                        c_new[i, 0] += dx
                        c_new[i, 1] += dy
                        s_new[i] = max(s_new[i] + ds, 0.02)
                        s = s_new[i]
                        c_new[i, 0] = np.clip(c_new[i, 0], s / 2, 1 - s / 2)
                        c_new[i, 1] = np.clip(c_new[i, 1], s / 2, 1 - s / 2)
                        if _is_feasible(c_new, s_new):
                            val = _sum_sides(s_new)
                            if val > best_obj + 1e-11:
                                best_centers, best_sides = c_new, s_new
                                best_obj = val
                                improved = True
        if not improved:
            break

    # final safety: ensure feasibility, else fall back to a known-good grid
    if not _is_feasible(best_centers, best_sides):
        c0, s0 = _repair(_grid(4)[0], _grid(4)[1])
        best_centers, best_sides = c0, s0

    return best_centers, best_sides


# EVOLVE-BLOCK-END
