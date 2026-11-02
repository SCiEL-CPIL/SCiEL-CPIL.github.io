"""Circle packing: Delaunay LP radii + equal-radius SLSQP + SA + polish."""
import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay

N = 21


def _delaunay_edges(xs, ys):
    """Return list of (i, j) edges from Delaunay triangulation."""
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
    """Solve LP for per-circle radii given fixed centers. Maximize sum(r)."""
    limits = np.minimum(np.minimum(xs, w - xs), np.minimum(ys, h - ys))
    if np.any(limits < 0):
        return np.zeros(N)

    A_list = []
    b_list = []
    edges = _delaunay_edges(xs, ys)
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
    """Largest r for single-radius packing feasibility."""
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


def _softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(x)))


def _spatial_hash_penalty(xs, ys, rs, beta=10.0, cell_size=None):
    """Spatial-hash accelerated smooth overlap penalty."""
    n = len(rs)
    if cell_size is None:
        max_r = np.max(rs)
        cell_size = max_r * 2.5
    grid = {}
    for i in range(n):
        ci = int(xs[i] // cell_size)
        cj = int(ys[i] // cell_size)
        key = (ci, cj)
        if key not in grid:
            grid[key] = []
        grid[key].append(i)

    penalty = 0.0
    for i in range(n):
        ci = int(xs[i] // cell_size)
        cj = int(ys[i] // cell_size)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                key = (ci + di, cj + dj)
                if key not in grid:
                    continue
                for j in grid[key]:
                    if j <= i:
                        continue
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist_sq = dx * dx + dy * dy
                    dist = np.sqrt(dist_sq)
                    min_dist = rs[i] + rs[j]
                    val = min_dist - dist
                    if val > 1.0 / beta:
                        penalty += val
                    else:
                        penalty += np.log1p(np.exp(beta * val)) / beta
    return penalty


def _make_hex(rows, w, h):
    """Hexagonal layout with given per-row counts, scaled to fit w x h."""
    pts = []
    for k, n in enumerate(rows):
        y = k * np.sqrt(3.0)
        off = 1.0 if k % 2 == 1 else 0.0
        for i in range(n):
            pts.append((i * 2.0 + off, y))
    pts = np.array(pts[:N])
    xs = pts[:, 0]
    ys = pts[:, 1]
    cx = 0.5 * (xs.max() + xs.min())
    cy = 0.5 * (ys.max() + ys.min())
    xs = xs - cx
    ys = ys - cy
    wc = xs.max() - xs.min()
    hc = ys.max() - ys.min()
    wt = wc + 2.0
    ht = hc + 2.0
    s = 2.0 / (wt + ht)
    xs_f = xs * s + 0.5 * (s * wt)
    ys_f = ys * s + 0.5 * (s * ht)
    return xs_f, ys_f


def _make_grid(rows, cols, w, h):
    """Grid layout scaled to fit w x h."""
    pts = []
    for j in range(rows):
        for i in range(cols):
            if len(pts) >= N:
                break
            pts.append((i * 2.0, j * 2.0))
    pts = np.array(pts[:N])
    xs = pts[:, 0]
    ys = pts[:, 1]
    cx = 0.5 * (xs.max() + xs.min())
    cy = 0.5 * (ys.max() + ys.min())
    xs = xs - cx
    ys = ys - cy
    wt = (xs.max() - xs.min()) + 2.0
    ht = (ys.max() - ys.min()) + 2.0
    s = 2.0 / (wt + ht)
    xs_f = xs * s + 0.5 * (s * wt)
    ys_f = ys * s + 0.5 * (s * ht)
    return xs_f, ys_f


def _make_random(w, h, seed):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.05, 0.95, N) * w
    ys = rng.uniform(0.05, 0.95, N) * h
    return xs, ys


def _check_valid(xs, ys, rs, w, h, tol=1e-6):
    """Check if configuration is valid (no overlaps, within bounds)."""
    if np.any(rs < 1e-9):
        return False
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx * dx + dy * dy
            min_dist = rs[i] + rs[j]
            if dist_sq < (min_dist - 1e-7) ** 2:
                return False
    return True


def _sa_optimize(w, h, xs, ys, max_iters=3000, seed=42, deadline=None):
    """Spatial-hash SA on centers with LP radii evaluation."""
    xs = xs.copy()
    ys = ys.copy()
    rs = _solve_lp_radii(xs, ys, w, h)
    r_max = min(w, h) / 2.0

    T = 0.02
    cooling = 0.9997
    beta = 15.0

    max_r = max(np.max(rs), 0.01)
    cell_size = max_r * 2.5
    current_val = -np.sum(rs)  # minimize negative sum

    rng = np.random.default_rng(seed)

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        idx = rng.integers(0, N)
        old_x, old_y = xs[idx], ys[idx]

        scale = T * 0.5
        xs[idx] += rng.normal(0, scale)
        ys[idx] += rng.normal(0, scale)

        # Clamp to bounds (use current radius as minimum distance from wall)
        r_i = rs[idx] if rs[idx] > 0 else 0.01
        xs[idx] = np.clip(xs[idx], r_i, w - r_i)
        ys[idx] = np.clip(ys[idx], r_i, h - r_i)

        # Recompute cell_size periodically
        if it % 300 == 0:
            max_r = max(np.max(rs), 0.01)
            cell_size = max_r * 2.5

        # Evaluate: use LP radii for new positions
        new_rs = _solve_lp_radii(xs, ys, w, h)
        new_val = -np.sum(new_rs)
        delta = new_val - current_val

        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_val = new_val
            rs = new_rs
        else:
            xs[idx], ys[idx] = old_x, old_y

        T *= cooling

    return xs, ys, rs


def _polish_lp(w, h, xs, ys, max_iter=500, deadline=None):
    """Polish centers via L-BFGS-B with smooth objective, then solve LP."""
    def objective(params):
        px = params[0::2]
        py = params[1::2]
        # Smooth penalty for overlaps and boundary violations
        pen = 0.0
        for i in range(N):
            # Boundary
            if px[i] < 0.01:
                pen += (0.01 - px[i]) ** 2 * 100
            if px[i] > w - 0.01:
                pen += (px[i] - (w - 0.01)) ** 2 * 100
            if py[i] < 0.01:
                pen += (0.01 - py[i]) ** 2 * 100
            if py[i] > h - 0.01:
                pen += (py[i] - (h - 0.01)) ** 2 * 100
        # Overlap penalty using estimated radii
        est_r = _solve_lp_radii(px, py, w, h)
        for i in range(N):
            for j in range(i + 1, N):
                dx = px[i] - px[j]
                dy = py[i] - py[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_d = est_r[i] + est_r[j]
                if dist < min_d:
                    pen += (min_d - dist) ** 2
        return pen - np.sum(est_r)

    x0 = np.zeros(2 * N)
    x0[0::2] = xs
    x0[1::2] = ys

    bounds = [(0.001, w - 0.001), (0.001, h - 0.001)] * N

    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12})
        if deadline is not None and time.time() > deadline:
            return xs, ys, _solve_lp_radii(xs, ys, w, h)
        px = res.x[0::2]
        py = res.x[1::2]
        pr = _solve_lp_radii(px, py, w, h)
        return px, py, pr
    except Exception:
        return xs, ys, _solve_lp_radii(xs, ys, w, h)


def _slsqp_equal_radius(w, h, xs, ys, r, deadline):
    """Equal-radius SLSQP jointly optimizing w and r."""
    if time.time() > deadline - 3.0:
        return w, h, r, xs, ys

    v0 = np.zeros(2 + 2 * N)
    v0[0] = w
    v0[1] = r
    v0[2:2 + N] = xs
    v0[2 + N:2 + 2 * N] = ys

    def objective(v):
        if v[0] <= 0.2 or (2.0 - v[0]) <= 0.2 or v[1] <= 0:
            return 1e6
        return -v[1]

    cons = [
        {'type': 'ineq', 'fun': lambda v: v[0] - 0.2},
        {'type': 'ineq', 'fun': lambda v: 2.0 - v[0] - 0.2},
    ]
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2 + i] - v[1]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - v[2 + i] - v[1]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2 + N + i] - v[1]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: (2.0 - v[0]) - v[2 + N + i] - v[1]})
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: np.sqrt((v[2 + i] - v[2 + j]) ** 2
                                                   + (v[2 + N + i] - v[2 + N + j]) ** 2) - 2.0 * v[1]
            })

    bounds = [(0.2, 1.8), (0.001, 1.0)] + [(0.0, 2.0)] * (2 * N)

    try:
        res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                       constraints=cons,
                       options={'maxiter': 100, 'ftol': 1e-9, 'disp': False})
    except Exception:
        return w, h, r, xs, ys

    if res.success:
        w_r = res.x[0]
        r_r = res.x[1]
        h_r = 2.0 - w_r
        xs_r = res.x[2:2 + N]
        ys_r = res.x[2 + N:2 + 2 * N]
        if w_r > 0 and h_r > 0 and r_r > 0:
            return w_r, h_r, r_r, xs_r, ys_r
    return w, h, r, xs, ys


def _scale_to_perimeter(result):
    """Scale result about bounding-box center so perimeter <= 4."""
    xs = result[:, 0]
    ys = result[:, 1]
    rs = result[:, 2]
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    bw = max_x - min_x
    bh = max_y - min_y
    perim = 2.0 * (bw + bh)
    if perim > 4.0 + 1e-9 and perim > 0:
        scale = 4.0 / perim
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        result[:, 0] = cx + (result[:, 0] - cx) * scale
        result[:, 1] = cy + (result[:, 1] - cy) * scale
        result[:, 2] = result[:, 2] * scale
    return result


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    best_value = -np.inf
    best_result = None

    def try_update(xs, ys, rs, w, h):
        nonlocal best_value, best_result
        if time.time() > deadline:
            return
        if not _check_valid(xs, ys, rs, w, h):
            return
        # Scale to ensure perimeter <= 4
        result = np.zeros((N, 3))
        result[:, 0] = xs
        result[:, 1] = ys
        result[:, 2] = rs
        result = _scale_to_perimeter(result)
        val = np.sum(result[:, 2])
        if val > best_value:
            best_value = val
            best_result = result.copy()

    # Try multiple aspect ratios and initial configurations
    widths = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50]

    # Hex row patterns that sum to 21
    hex_patterns = [
        [4, 5, 4, 4, 4],       # 21
        [5, 4, 5, 4, 3],       # 21
        [3, 5, 4, 5, 4],       # 21
        [4, 4, 5, 4, 4],       # 21
        [5, 5, 4, 3, 4],       # 21
        [4, 5, 5, 3, 4],       # 21
    ]

    grid_patterns = [
        (4, 5),  # 20, add 1
        (5, 4),  # 20, add 1
        (3, 7),  # 21
        (7, 3),  # 21
        (2, 11), # 22, use 21
        (11, 2), # 22, use 21
    ]

    for w in widths:
        if time.time() > deadline - 5:
            break
        h = 2.0 - w

        # Hex initializations
        for pat in hex_patterns:
            if time.time() > deadline - 5:
                break
            try:
                xs, ys = _make_hex(pat, w, h)
                # Clip to bounds
                xs = np.clip(xs, 0.01, w - 0.01)
                ys = np.clip(ys, 0.01, h - 0.01)

                # LP radii
                rs = _solve_lp_radii(xs, ys, w, h)
                try_update(xs, ys, rs, w, h)

                # SA refinement
                if time.time() < deadline - 15:
                    xs_sa, ys_sa, rs_sa = _sa_optimize(w, h, xs, ys, max_iters=2000, seed=42 + int(w * 100), deadline=deadline - 10)
                    try_update(xs_sa, ys_sa, rs_sa, w, h)

                    # Polish
                    if time.time() < deadline - 8:
                        xs_p, ys_p, rs_p = _polish_lp(w, h, xs_sa, ys_sa, max_iter=300, deadline=deadline - 5)
                        try_update(xs_p, ys_p, rs_p, w, h)
            except Exception:
                pass

        # Grid initializations
        for rows, cols in grid_patterns:
            if time.time() > deadline - 5:
                break
            try:
                xs, ys = _make_grid(rows, cols, w, h)
                xs = np.clip(xs, 0.01, w - 0.01)
                ys = np.clip(ys, 0.01, h - 0.01)

                rs = _solve_lp_radii(xs, ys, w, h)
                try_update(xs, ys, rs, w, h)

                if time.time() < deadline - 12:
                    xs_sa, ys_sa, rs_sa = _sa_optimize(w, h, xs, ys, max_iters=1500, seed=200 + int(w * 100), deadline=deadline - 8)
                    try_update(xs_sa, ys_sa, rs_sa, w, h)
            except Exception:
                pass

        # Random initialization
        if time.time() < deadline - 15:
            try:
                xs, ys = _make_random(w, h, seed=777 + int(w * 100))
                rs = _solve_lp_radii(xs, ys, w, h)
                try_update(xs, ys, rs, w, h)

                xs_sa, ys_sa, rs_sa = _sa_optimize(w, h, xs, ys, max_iters=2000, seed=500 + int(w * 100), deadline=deadline - 10)
                try_update(xs_sa, ys_sa, rs_sa, w, h)

                if time.time() < deadline - 8:
                    xs_p, ys_p, rs_p = _polish_lp(w, h, xs_sa, ys_sa, max_iter=300, deadline=deadline - 5)
                    try_update(xs_p, ys_p, rs_p, w, h)
            except Exception:
                pass

        # Equal-radius SLSQP on best config for this w
        if time.time() < deadline - 8:
            try:
                # Use a reasonable initial equal radius
                r_init = min(w, h) / (2 * 4)  # rough estimate
                xs_eq, ys_eq = _make_hex([4, 5, 4, 4, 4], w, h)
                xs_eq = np.clip(xs_eq, 0.01, w - 0.01)
                ys_eq = np.clip(ys_eq, 0.01, h - 0.01)
                r_eq = _max_equal_radius(xs_eq, ys_eq, w, h)
                if r_eq > 0.001:
                    w_r, h_r, r_r, xs_r, ys_r = _slsqp_equal_radius(w, h, xs_eq, ys_eq, r_eq, deadline)
                    if r_r > 0:
                        rs_eq = np.full(N, r_r)
                        try_update(xs_r, ys_r, rs_eq, w_r, h_r)
            except Exception:
                pass

    # Final polish on best result
    if best_result is not None and time.time() < deadline - 3:
        try:
            xs = best_result[:, 0].copy()
            ys = best_result[:, 1].copy()
            rs = best_result[:, 2].copy()
            # Estimate w, h from bounding box
            min_x = np.min(xs - rs)
            max_x = np.max(xs + rs)
            min_y = np.min(ys - rs)
            max_y = np.max(ys + rs)
            w_est = max_x - min_x
            h_est = max_y - min_y
            # Shift to origin
            xs_shift = xs - min_x
            ys_shift = ys - min_y

            xs_p, ys_p, rs_p = _polish_lp(w_est, h_est, xs_shift, ys_shift, max_iter=400, deadline=deadline - 1)
            # Shift back
            xs_p += min_x
            ys_p += min_y
            try_update(xs_p, ys_p, rs_p, w_est, h_est)
        except Exception:
            pass

    # Fallback
    if best_result is None:
        w, h = 1.4, 0.6
        a, b = 7, 3
        r = 0.08
        best_result = np.zeros((N, 3))
        for i in range(N):
            col = i % a
            row = i // a
            best_result[i, 0] = (col + 0.5) * w / a
            best_result[i, 1] = (row + 0.5) * h / b
            best_result[i, 2] = r
        best_result = _scale_to_perimeter(best_result)

    return best_result
