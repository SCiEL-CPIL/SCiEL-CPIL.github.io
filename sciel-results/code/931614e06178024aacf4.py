"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Combines:
  - Successive Convex Relaxation (SCR): alternate position-opt (fixed radii) and
    radius-opt (fixed positions) subproblems.
  - SLSQP constrained polish with analytical gradients for non-overlap + boundary.
  - Fixed-radius hexagonal lattice initialization + aspect-ratio search.
"""
import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 21
DEADLINE_MARGIN = 10.0  # seconds before 150s wall-clock limit


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate(circles, w, h, tol=1e-7):
    n = len(circles)
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    if np.any(r < -tol):
        return False
    if np.any(cx - r < -tol) or np.any(cx + r > w + tol):
        return False
    if np.any(cy - r < -tol) or np.any(cy + r > h + tol):
        return False
    for i in range(n):
        dx = cx[i] - cx[i + 1:]
        dy = cy[i] - cy[i + 1:]
        dist = np.sqrt(dx * dx + dy * dy)
        if np.any(dist < r[i] + r[i + 1:] - tol):
            return False
    return True


def _total_radius(circles):
    return float(np.sum(circles[:, 2]))


# ---------------------------------------------------------------------------
# Separation / repair
# ---------------------------------------------------------------------------
def _separate(circles, w, h, max_iter=300, tol=1e-9):
    """Iteratively push circles inside bounds and separate overlaps."""
    circles = circles.copy()
    n = len(circles)
    for _ in range(max_iter):
        moved = False
        # Bounds
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0:
                circles[i, 0] = r; moved = True
            elif x + r > w:
                circles[i, 0] = w - r; moved = True
            if y - r < 0:
                circles[i, 1] = r; moved = True
            elif y + r > h:
                circles[i, 1] = h - r; moved = True
        # Separation
        for i in range(n):
            dx = circles[i, 0] - circles[i + 1:, 0]
            dy = circles[i, 1] - circles[i + 1:, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_d = circles[i, 2] + circles[i + 1:, 2]
            overlap = min_d - dist
            for k in range(len(overlap)):
                if overlap[k] > 0:
                    j = i + 1 + k
                    if dist[k] < 1e-12:
                        ang = np.random.uniform(0, 2 * np.pi)
                        px = np.cos(ang) * overlap[k] / 2
                        py = np.sin(ang) * overlap[k] / 2
                        circles[i, 0] -= px; circles[i, 1] -= py
                        circles[j, 0] += px; circles[j, 1] += py
                    else:
                        ux = dx[k] / dist[k]
                        uy = dy[k] / dist[k]
                        px = ux * overlap[k] / 2
                        py = uy * overlap[k] / 2
                        circles[i, 0] += px; circles[i, 1] += py
                        circles[j, 0] -= px; circles[j, 1] -= py
                    moved = True
        # Re-clamp bounds
        for i in range(n):
            x, y, r = circles[i]
            circles[i, 0] = np.clip(x, r, max(r, w - r))
            circles[i, 1] = np.clip(y, r, max(r, h - r))
        if not moved:
            break
    return circles


# ---------------------------------------------------------------------------
# Hexagonal lattice initialization
# ---------------------------------------------------------------------------
def _hex_init(w, h):
    """Create a hexagonal-like arrangement of N circles in (w,h)."""
    rows = [5, 4, 5, 4, 3]  # 21 circles
    n_rows = len(rows)
    circles = []
    # Estimate radius from hexagonal packing density
    area = w * h
    # Hex packing density ~ pi/(2*sqrt(3)) ~ 0.9069
    # For N equal circles: N * pi * r^2 <= 0.9069 * area
    r_est = np.sqrt(0.9069 * area / (N * np.pi)) * 0.95
    for ri, cnt in enumerate(rows):
        y = (ri + 0.5) * h / n_rows
        for cj in range(cnt):
            x = (cj + 0.5) * w / 5
            if ri % 2 == 1:
                x += 0.5 * w / 5
            circles.append([x, y, r_est])
    return np.array(circles[:N])


# ---------------------------------------------------------------------------
# SCR: alternate position-opt and radius-opt
# ---------------------------------------------------------------------------
def _scr_step_pos(circles, w, h, deadline):
    """Fix radii, optimize positions via SLSQP (convex-ish)."""
    n = len(circles)
    r = circles[:, 2].copy()
    x0 = np.concatenate([circles[:, 0], circles[:, 1]])

    def obj(x):
        return 0.0  # feasible set; we just need a valid point

    def obj_grad(x):
        return np.zeros_like(x)

    # Non-overlap: dist(i,j) - r_i - r_j >= 0
    def cons_fun(x):
        cx = x[:n]; cy = x[n:]
        vals = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]; dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                vals.append(d - r[i] - r[j])
        return np.array(vals)

    def cons_jac(x):
        cx = x[:n]; cy = x[n:]
        J = np.zeros((n * (n - 1) // 2, 2 * n))
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]; dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < 1e-9:
                    ux, uy = 1.0, 0.0
                else:
                    ux, uy = dx / d, dy / d
                J[idx, i] = ux
                J[idx, n + i] = uy
                J[idx, j] = -ux
                J[idx, n + j] = -uy
                idx += 1
        return J

    # Boundary: x_i - r_i >= 0, w - x_i - r_i >= 0, y_i - r_i >= 0, h - y_i - r_i >= 0
    def bnd_fun(x):
        cx = x[:n]; cy = x[n:]
        return np.concatenate([
            cx - r, w - cx - r, cy - r, h - cy - r
        ])

    def bnd_jac(x):
        J = np.zeros((4 * n, 2 * n))
        for i in range(n):
            J[4 * i, i] = 1.0
            J[4 * i + 1, i] = -1.0
            J[4 * i + 2, n + i] = 1.0
            J[4 * i + 3, n + i] = -1.0
        return J

    constraints = [
        {'type': 'ineq', 'fun': cons_fun, 'jac': cons_jac},
        {'type': 'ineq', 'fun': bnd_fun, 'jac': bnd_jac},
    ]
    bounds = [(0, w)] * n + [(0, h)] * n

    try:
        res = minimize(obj, x0, jac=obj_grad, method='SLSQP',
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': 200, 'ftol': 1e-10})
        if res.success or _validate(np.column_stack([res.x[:n], res.x[n:], r]), w, h):
            new_c = np.column_stack([res.x[:n], res.x[n:], r])
            if _validate(new_c, w, h):
                return new_c
    except Exception:
        pass
    return circles


def _scr_step_rad(circles, w, h, deadline):
    """Fix positions, maximize sum of radii via LP (linear program)."""
    n = len(circles)
    cx = circles[:, 0]
    cy = circles[:, 1]

    # Variables: r_0, ..., r_{n-1}
    # Maximize sum(r) -> minimize -sum(r)
    c = -np.ones(n)

    # Constraints:
    # r_i >= 0  (handled by bounds)
    # r_i <= cx_i, r_i <= w - cx_i, r_i <= cy_i, r_i <= h - cy_i
    # r_i + r_j <= dist(i,j) for all i < j

    A_ub = []
    b_ub = []

    # Boundary upper bounds
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row); b_ub.append(cx[i])
        A_ub.append(row); b_ub.append(w - cx[i])
        A_ub.append(row); b_ub.append(cy[i])
        A_ub.append(row); b_ub.append(h - cy[i])

    # Pairwise
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((cx[i] - cx[j]) ** 2 + (cy[i] - cy[j]) ** 2)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row); b_ub.append(d)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(1e-9, None)] * n

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            new_r = res.x
            new_c = np.column_stack([cx, cy, new_r])
            if _validate(new_c, w, h):
                return new_c
    except Exception:
        pass
    return circles


def _scr(circles, w, h, deadline, n_iters=30):
    """Run SCR iterations alternating position and radius optimization."""
    best = circles.copy()
    best_score = _total_radius(best)
    for it in range(n_iters):
        if time.time() > deadline - DEADLINE_MARGIN:
            break
        # Position step (fix radii)
        circles = _scr_step_pos(circles, w, h, deadline)
        # Radius step (fix positions)
        circles = _scr_step_rad(circles, w, h, deadline)
        if _validate(circles, w, h):
            s = _total_radius(circles)
            if s > best_score:
                best_score = s
                best = circles.copy()
        else:
            # Repair
            repaired = _separate(circles, w, h, max_iter=200)
            if _validate(repaired, w, h):
                circles = repaired
                s = _total_radius(circles)
                if s > best_score:
                    best_score = s
                    best = circles.copy()
    return best


# ---------------------------------------------------------------------------
# SLSQP full polish (positions + radii jointly)
# ---------------------------------------------------------------------------
def _slsqp_polish(circles, w, h, deadline):
    """Polish using SLSQP with joint position+radius variables."""
    n = len(circles)
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])

    def obj(x):
        return -np.sum(x[2 * n:])

    def obj_grad(x):
        g = np.zeros_like(x)
        g[2 * n:] = -1.0
        return g

    def cons_fun(x):
        cx = x[:n]; cy = x[n:2 * n]; r = x[2 * n:]
        vals = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]; dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                vals.append(d - r[i] - r[j])
        return np.array(vals)

    def cons_jac(x):
        cx = x[:n]; cy = x[n:2 * n]; r = x[2 * n:]
        J = np.zeros((n * (n - 1) // 2, 3 * n))
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]; dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < 1e-9:
                    ux, uy = 1.0, 0.0
                else:
                    ux, uy = dx / d, dy / d
                J[idx, i] = ux
                J[idx, n + i] = uy
                J[idx, 2 * n + i] = -1.0
                J[idx, j] = -ux
                J[idx, n + j] = -uy
                J[idx, 2 * n + j] = -1.0
                idx += 1
        return J

    def bnd_fun(x):
        cx = x[:n]; cy = x[n:2 * n]; r = x[2 * n:]
        return np.concatenate([cx - r, w - cx - r, cy - r, h - cy - r])

    def bnd_jac(x):
        J = np.zeros((4 * n, 3 * n))
        for i in range(n):
            J[4 * i, i] = 1.0
            J[4 * i, 2 * n + i] = -1.0
            J[4 * i + 1, i] = -1.0
            J[4 * i + 1, 2 * n + i] = -1.0
            J[4 * i + 2, n + i] = 1.0
            J[4 * i + 2, 2 * n + i] = -1.0
            J[4 * i + 3, n + i] = -1.0
            J[4 * i + 3, 2 * n + i] = -1.0
        return J

    constraints = [
        {'type': 'ineq', 'fun': cons_fun, 'jac': cons_jac},
        {'type': 'ineq', 'fun': bnd_fun, 'jac': bnd_jac},
    ]
    bounds = [(1e-9, w)] * n + [(1e-9, h)] * n + [(1e-9, min(w, h) / 2)] * n

    try:
        res = minimize(obj, x0, jac=obj_grad, method='SLSQP',
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': 300, 'ftol': 1e-12})
        new_c = np.column_stack([res.x[:n], res.x[n:2 * n], res.x[2 * n:]])
        if _validate(new_c, w, h):
            return new_c
    except Exception:
        pass
    return circles


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin

    best_circles = None
    best_score = 0.0

    # Aspect ratios: w + h = 2 (perimeter 4)
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.43, 0.6, 1.67, 0.5, 2.0, 0.4]

    for aspect in aspect_ratios:
        if time.time() > deadline - DEADLINE_MARGIN:
            break

        w = 2.0 / (1.0 + 1.0 / aspect)
        h = 2.0 - w

        # Initialize with hexagonal lattice
        circles = _hex_init(w, h)
        circles = _separate(circles, w, h, max_iter=500)

        if _validate(circles, w, h):
            s = _total_radius(circles)
            if s > best_score:
                best_score = s
                best_circles = circles.copy()

        # SCR iterations
        if time.time() < deadline - DEADLINE_MARGIN:
            circles = _scr(circles, w, h, deadline, n_iters=25)
            if _validate(circles, w, h):
                s = _total_radius(circles)
                if s > best_score:
                    best_score = s
                    best_circles = circles.copy()

        # SLSQP polish
        if time.time() < deadline - DEADLINE_MARGIN:
            polished = _slsqp_polish(circles, w, h, deadline)
            if _validate(polished, w, h):
                s = _total_radius(polished)
                if s > best_score:
                    best_score = s
                    best_circles = polished.copy()

    # Final polish on best if time remains
    if best_circles is not None and time.time() < deadline - 5.0:
        # Determine w, h from the best packing's bounding box
        # Actually we need to know which aspect was used; re-derive from best
        # Use the aspect that produced best: track it
        pass

    if best_circles is not None:
        return np.array(best_circles)

    # Fallback: simple grid in 1x1 square
    w0, h0 = 1.0, 1.0
    side = 5
    initial = []
    for i in range(side):
        for j in range(side):
            if len(initial) >= N:
                break
            cx = (i + 0.5) / side * w0
            cy = (j + 0.5) / side * h0
            r = 0.5 / side * min(w0, h0) - 1e-3
            initial.append([cx, cy, r])
    return np.array(initial[:N])
