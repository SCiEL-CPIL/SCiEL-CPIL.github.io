import numpy as np
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay
import time

N = 21

def _delaunay_edges(xs, ys):
    """Return list of (i, j) edges from Delaunay triangulation of centers."""
    points = np.column_stack((xs, ys))
    try:
        tri = Delaunay(points)
    except Exception:
        return None
    edges = set()
    for simplex in tri.simplices:
        edges.add(tuple(sorted((simplex[0], simplex[1]))))
        edges.add(tuple(sorted((simplex[1], simplex[2]))))
        edges.add(tuple(sorted((simplex[0], simplex[2]))))
    return list(edges)

def _solve_lp_radii(xs, ys, w, h):
    """
    Solve LP for per-circle radii given fixed centers and rectangle w x h.
    Maximize sum(r) s.t.  r_i + r_j <= d_ij  (Delaunay edges)
                          r_i <= min(x_i, w-x_i, y_i, h-y_i)
                          r_i >= 0
    """
    limits = np.minimum(np.minimum(xs, w - xs), np.minimum(ys, h - ys))
    if np.any(limits < 0):
        return np.zeros(N)
    limits = np.maximum(limits, 0.0)

    edges = _delaunay_edges(xs, ys)
    A_list = []
    b_list = []
    if edges is not None:
        for i, j in edges:
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            row = np.zeros(N)
            row[i] = 1
            row[j] = 1
            A_list.append(row)
            b_list.append(dist)

    A_bound = np.eye(N)
    if A_list:
        A_ub = np.vstack([A_list, A_bound])
        b_ub = np.concatenate([b_list, limits])
    else:
        A_ub = A_bound
        b_ub = limits

    c = -np.ones(N)
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        if res.success:
            return np.maximum(0.0, res.x)
    except Exception:
        pass
    return np.maximum(0.0, limits * 0.9)

def _max_equal_radius(xs, ys, w, h):
    """Largest single radius r feasible for these centers in w x h rectangle."""
    limits = np.minimum(np.minimum(xs, w - xs), np.minimum(ys, h - ys))
    r = float(np.min(limits))
    if r < 0:
        return 0.0
    edges = _delaunay_edges(xs, ys)
    if edges is not None:
        for i, j in edges:
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            r = min(r, 0.5 * np.sqrt(dx * dx + dy * dy))
    return max(0.0, r)

def _make_hex(rows):
    """Hexagonal layout, scaled to perimeter-4 (w+h=2). Returns xs, ys, w, h."""
    pts = []
    for k, n in enumerate(rows):
        y = k * np.sqrt(3.0)
        off = 1.0 if k % 2 == 1 else 0.0
        for i in range(n):
            pts.append((i * 2.0 + off, y))
    pts = np.array(pts[:N])
    xs = pts[:, 0]; ys = pts[:, 1]
    cx = 0.5 * (xs.max() + xs.min()); cy = 0.5 * (ys.max() + ys.min())
    xs = xs - cx; ys = ys - cy
    wt = (xs.max() - xs.min()) + 2.0
    ht = (ys.max() - ys.min()) + 2.0
    s = 2.0 / (wt + ht)
    xs_f = xs * s + 0.5 * (s * wt)
    ys_f = ys * s + 0.5 * (s * ht)
    return xs_f, ys_f, s * wt, s * ht

def _make_grid(rows, cols):
    pts = []
    for j in range(rows):
        for i in range(cols):
            if len(pts) >= N:
                break
            pts.append((i * 2.0, j * 2.0))
    pts = np.array(pts[:N])
    xs = pts[:, 0]; ys = pts[:, 1]
    cx = 0.5 * (xs.max() + xs.min()); cy = 0.5 * (ys.max() + ys.min())
    xs = xs - cx; ys = ys - cy
    wt = (xs.max() - xs.min()) + 2.0
    ht = (ys.max() - ys.min()) + 2.0
    s = 2.0 / (wt + ht)
    xs_f = xs * s + 0.5 * (s * wt)
    ys_f = ys * s + 0.5 * (s * ht)
    return xs_f, ys_f, s * wt, s * ht

def _physics_settle(w, h, r_target, seed, deadline):
    """
    Settle N particles in w x h rectangle via velocity-Verlet with a
    short-range repulsive (Lennard-Jones-like) potential + soft boundary.
    Returns xs, ys (settle into dense, non-overlapping layout).
    """
    rng = np.random.RandomState(seed)
    xs = rng.uniform(0.15, w - 0.15, N)
    ys = rng.uniform(0.15, h - 0.15, N)
    vx = np.zeros(N); vy = np.zeros(N)

    # repulsion range ~ 2 * r_target
    sigma = 2.0 * r_target
    dt = 0.002
    n_steps = 2500

    def forces(x, y):
        fx = np.zeros(N); fy = np.zeros(N)
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]; dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                d = np.sqrt(d2) if d2 > 1e-12 else 1e-6
                # LJ-like repulsion: strong at close range, decays
                sr = sigma / d
                if sr > 1.0:
                    # (sr^12 - sr^6) repulsive core, scaled
                    f = 12.0 * (sr ** 12 - 0.5 * sr ** 6)
                    if f < 0:
                        f = 0.0
                    fx[i] += f * dx / d
                    fy[i] += f * dy / d
                    fx[j] -= f * dx / d
                    fy[j] -= f * dy / d
        # soft boundary: push inward near walls
        margin = 0.02
        for i in range(N):
            if x[i] < margin:
                fx[i] += (margin - x[i]) * 50.0
            if x[i] > w - margin:
                fx[i] -= (x[i] - (w - margin)) * 50.0
            if y[i] < margin:
                fy[i] += (margin - y[i]) * 50.0
            if y[i] > h - margin:
                fy[i] -= (y[i] - (h - margin)) * 50.0
        return fx, fy

    for step in range(n_steps):
        if time.time() > deadline - 2.0:
            break
        fx, fy = forces(xs, ys)
        vx = 0.98 * vx + fx * dt
        vy = 0.98 * vy + fy * dt
        xs = xs + vx * dt
        ys = ys + vy * dt
        # hard clamp
        xs = np.clip(xs, 0.01, w - 0.01)
        ys = np.clip(ys, 0.01, h - 0.01)
    return xs, ys

def _refine_slsqp(w, h, xs, ys, r, deadline, maxiter=150):
    """
    Equal-radius SLSQP refinement. Variables: [w, r, x0..x20, y0..y20], h=2-w.
    Maximize r s.t. boundary + pairwise non-overlap (squared-distance for smoothness).
    """
    if time.time() > deadline - 3.0:
        return w, h, r, xs, ys

    v0 = np.zeros(2 + 2 * N)
    v0[0] = w
    v0[1] = r
    v0[2:2 + N] = xs
    v0[2 + N:2 + 2 * N] = ys

    def objective(v):
        wv = v[0]; rv = v[1]; hv = 2.0 - wv
        if wv <= 0.1 or hv <= 0.1 or rv <= 0:
            return 1e6
        return -rv

    def constraints(v):
        wv = v[0]; rv = v[1]; hv = 2.0 - wv
        xv = v[2:2 + N]; yv = v[2 + N:2 + 2 * N]
        cons = []
        cons.append({'type': 'ineq', 'fun': lambda v: v[0] - 0.2})
        cons.append({'type': 'ineq', 'fun': lambda v: 2.0 - v[0] - 0.2})
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2 + i] - v[1]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - v[2 + i] - v[1]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2 + N + i] - v[1]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: (2.0 - v[0]) - v[2 + N + i] - v[1]})
        for i in range(N):
            for j in range(i + 1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: ((v[2 + i] - v[2 + j]) ** 2 + (v[2 + N + i] - v[2 + N + j]) ** 2) - 4.0 * (v[1] ** 2)
                })
        return cons

    bounds = [(0.2, 1.8), (0.001, 1.0)] + [(0.0, 2.0)] * (2 * N)

    try:
        res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-10, 'disp': False})
    except Exception:
        return w, h, r, xs, ys

    if res.success or res.fun < -r:
        w_r = res.x[0]; r_r = res.x[1]; h_r = 2.0 - w_r
        xs_r = res.x[2:2 + N]; ys_r = res.x[2 + N:2 + 2 * N]
        if w_r > 0 and h_r > 0 and r_r > 0:
            # validate
            ok = True
            for i in range(N):
                if xs_r[i] < r_r - 1e-6 or xs_r[i] > w_r - r_r + 1e-6:
                    ok = False; break
                if ys_r[i] < r_r - 1e-6 or ys_r[i] > h_r - r_r + 1e-6:
                    ok = False; break
            if ok:
                for i in range(N):
                    for j in range(i + 1, N):
                        d2 = (xs_r[i] - xs_r[j]) ** 2 + (ys_r[i] - ys_r[j]) ** 2
                        if d2 < 4.0 * (r_r ** 2) - 1e-6:
                            ok = False; break
                    if not ok:
                        break
            if ok:
                return w_r, h_r, r_r, xs_r, ys_r
    return w, h, r, xs, ys

def _scale_to_perimeter(result):
    """Scale result about bounding-box center so perimeter <= 4."""
    xs = result[:, 0]; ys = result[:, 1]; rs = result[:, 2]
    min_x = np.min(xs - rs); max_x = np.max(xs + rs)
    min_y = np.min(ys - rs); max_y = np.max(ys + rs)
    bw = max_x - min_x; bh = max_y - min_y
    perim = 2.0 * (bw + bh)
    if perim > 4.0 + 1e-9 and perim > 0:
        scale = 4.0 / perim
        cx = 0.5 * (min_x + max_x); cy = 0.5 * (min_y + max_y)
        for i in range(N):
            result[i, 0] = cx + (result[i, 0] - cx) * scale
            result[i, 1] = cy + (result[i, 1] - cy) * scale
            result[i, 2] = rs[i] * scale
    return result

def _make_result_lp(w, h, xs, ys):
    rs = _solve_lp_radii(xs, ys, w, h)
    result = np.zeros((N, 3))
    result[:, 0] = xs
    result[:, 1] = ys
    result[:, 2] = rs
    result[:, 2] = np.maximum(result[:, 2], 0.0)
    _scale_to_perimeter(result)
    return result

def _make_result_equal(w, h, xs, ys, r):
    result = np.zeros((N, 3))
    result[:, 0] = xs
    result[:, 1] = ys
    result[:, 2] = r
    _scale_to_perimeter(result)
    return result

def _sum_radii(result):
    return float(np.sum(result[:, 2]))

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin for 150s limit

    best_result = None
    best_sum = 0.0

    def consider(result):
        nonlocal best_result, best_sum
        if result is None:
            return
        s = _sum_radii(result)
        if s > best_sum:
            best_sum = s
            best_result = result.copy()

    # ---- Build candidate center layouts ----
    layouts = []

    # Hexagonal variations
    hex_rows = [
        [7, 7, 7],
        [4, 5, 4, 4, 4],
        [3, 4, 4, 4, 4, 2],
        [5, 5, 5, 3, 3],
        [6, 6, 5, 4],
        [8, 7, 6],
        [9, 6, 6],
        [10, 5, 6],
    ]
    for rows in hex_rows:
        if sum(rows) >= N:
            try:
                xs, ys, w, h = _make_hex(rows)
                layouts.append((xs, ys, w, h))
            except Exception:
                pass

    # Grid variations
    for rows, cols in [(5, 5), (3, 7), (4, 6), (7, 3), (6, 4), (2, 11), (11, 2), (7, 4)]:
        try:
            xs, ys, w, h = _make_grid(rows, cols)
            layouts.append((xs, ys, w, h))
        except Exception:
            pass

    # Physics-settled layouts (fast dense packings) from several seeds and aspect ratios
    for (w, h) in [(1.0, 1.0), (1.2, 0.8), (0.8, 1.2), (1.3, 0.7), (1.1, 0.9)]:
        for seed in range(4):
            if time.time() > deadline - 5.0:
                break
            r_target = 0.085
            try:
                xs, ys = _physics_settle(w, h, r_target, seed, deadline)
                layouts.append((xs, ys, w, h))
            except Exception:
                pass

    # Random layouts
    rng = np.random.RandomState(0)
    for _ in range(6):
        if time.time() > deadline - 5.0:
            break
        w = rng.uniform(0.8, 1.2)
        h = 2.0 - w
        xs = rng.uniform(0.1, w - 0.1, N)
        ys = rng.uniform(0.1, h - 0.1, N)
        layouts.append((xs, ys, w, h))

    # ---- For each layout: LP radii (variable) + SLSQP center refinement (equal) ----
    for (xs, ys, w, h) in layouts:
        if time.time() > deadline - 4.0:
            break
        # 1) Direct LP radii for these centers (variable radius, good baseline)
        try:
            res_lp = _make_result_lp(w, h, xs, ys)
            consider(res_lp)
        except Exception:
            pass

        # 2) Equal-radius SLSQP refinement to improve centers, then LP radii
        if time.time() > deadline - 4.0:
            break
        try:
            r_eq = _max_equal_radius(xs, ys, w, h)
            if r_eq <= 0:
                r_eq = 0.05
            w2, h2, r2, xs2, ys2 = _refine_slsqp(w, h, xs, ys, r_eq, deadline, maxiter=120)
            # Equal-radius result
            res_eq = _make_result_equal(w2, h2, xs2, ys2, r2)
            consider(res_eq)
            # LP radii on refined centers (usually better than equal)
            res_lp2 = _make_result_lp(w2, h2, xs2, ys2)
            consider(res_lp2)
        except Exception:
            pass

    # ---- Final high-precision polish on the best center layout ----
    if best_result is not None and time.time() < deadline - 8.0:
        # Reconstruct best centers from the LP result before scaling is tricky;
        # instead do a final SLSQP polish on the best-equal-radius config found.
        # We keep the best LP result as the answer; do one more LP on best centers.
        pass

    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
        return np.array(cells)

    # Final safety: ensure perimeter <= 4 and non-negative radii
    best_result[:, 2] = np.maximum(best_result[:, 2], 0.0)
    _scale_to_perimeter(best_result)
    return best_result
