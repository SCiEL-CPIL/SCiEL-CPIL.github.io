"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Combines:
 - Delaunay-based LP radii solver (exact feasibility, decoupled structure)
 - equal-radius SLSQP refinement jointly optimizing aspect ratio w and radius r
 - hexagonal / grid / random initial configurations
 - Smooth gradient-based polish (L-BFGS-B with softplus penalties) for per-circle radii
 - Vectorized validity checks for speed
"""
import numpy as np
import time
import random
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay

N = 21


def _softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(x)))


def _sigmoid(x):
    return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(-x)))


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
    Solve LP for per-circle radii given fixed centers and rectangle.
    Maximize sum(r) s.t.  r_i + r_j <= d_ij  (Delaunay edges)
                          r_i <= min(x_i, w-x_i, y_i, h-y_i)
                          r_i >= 0
    Returns array of radii (zeros on failure).
    """
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
    """
    Largest r such that a single-radius packing is feasible:
    r <= min_i (x_i, w-x_i, y_i, h-y_i)
    r <= d_ij / 2 for all Delaunay edges (i, j).
    """
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


def _make_hex(rows, w, h):
    """Hexagonal layout of given per-row counts, scaled to fit w x h."""
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
    return xs_f, ys_f, s * wt, s * ht


def _make_grid(rows, cols, w, h):
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
    return xs_f, ys_f, s * wt, s * ht


def _make_random(w, h, seed):
    rng = np.random.RandomState(seed)
    xs = rng.uniform(0.1, 0.9, N) * w
    ys = rng.uniform(0.1, 0.9, N) * h
    return xs, ys, w, h


def _refine_slsqp(w, h, xs, ys, r, deadline):
    """
    Equal-radius SLSQP refinement.
    Variables: [w, r, x0..x20, y0..y20].  h = 2 - w.
    Maximize r  s.t.  boundary + pairwise non-overlap.
    Returns (w, h, r, xs, ys) or the input on failure.
    """
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

    def make_cons():
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
        return cons

    bounds = [(0.2, 1.8), (0.001, 1.0)] + [(0.0, 2.0)] * (2 * N)

    try:
        res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                       constraints=make_cons(),
                       options={'maxiter': 120, 'ftol': 1e-9, 'disp': False})
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


def _polish_gradient(w, h, xs, ys, rs, deadline, max_iter=500):
    """
    Smooth gradient-based polish using L-BFGS-B with softplus penalty.
    Variables: [w, x0..x20, y0..y20, r0..r20]. h = 2 - w.
    Maximizes sum(r_i) subject to boundary and non-overlap via softplus penalties.
    Returns (w, h, xs, ys, rs) or input on failure.
    """
    if time.time() > deadline - 5.0:
        return w, h, xs, ys, rs

    # State vector: [w, x0..x20, y0..y20, r0..r20]
    v0 = np.zeros(1 + 3 * N)
    v0[0] = w
    v0[1:1 + N] = xs
    v0[1 + N:1 + 2 * N] = ys
    v0[1 + 2 * N:1 + 3 * N] = rs

    idx_i, idx_j = np.triu_indices(N, k=1)

    def objective_and_grad(v):
        w_val = v[0]
        h_val = 2.0 - w_val
        x = v[1:1 + N]
        y = v[1 + N:1 + 2 * N]
        r = v[1 + 2 * N:1 + 3 * N]

        # Ensure positive dimensions
        if w_val < 0.05 or h_val < 0.05:
            return 1e6, np.zeros_like(v)

        obj = -np.sum(r)
        grad = np.zeros_like(v)
        grad[1 + 2 * N:1 + 3 * N] = -1.0  # d(-sum_r)/dr_i

        # Softplus penalty parameters
        eps_b = 1e-3  # boundary
        eps_o = 1e-3  # overlap

        # Boundary violations
        # Left: r_i - x_i
        v_l = r - x
        sp_l = eps_b * np.log1p(np.exp(np.clip(v_l / eps_b, -20, 20)))
        obj += 500.0 * np.sum(sp_l)
        sig_l = _sigmoid(v_l / eps_b)
        grad[1 + 2 * N:1 + 3 * N] += 500.0 * sig_l
        grad[1:1 + N] -= 500.0 * sig_l

        # Right: r_i - (w - x_i)
        v_r = r - (w_val - x)
        sp_r = eps_b * np.log1p(np.exp(np.clip(v_r / eps_b, -20, 20)))
        obj += 500.0 * np.sum(sp_r)
        sig_r = _sigmoid(v_r / eps_b)
        grad[1 + 2 * N:1 + 3 * N] += 500.0 * sig_r
        grad[0] += 500.0 * np.sum(sig_r)
        grad[1:1 + N] -= 500.0 * sig_r

        # Bottom: r_i - y_i
        v_b = r - y
        sp_b = eps_b * np.log1p(np.exp(np.clip(v_b / eps_b, -20, 20)))
        obj += 500.0 * np.sum(sp_b)
        sig_b = _sigmoid(v_b / eps_b)
        grad[1 + 2 * N:1 + 3 * N] += 500.0 * sig_b
        grad[1 + N:1 + 2 * N] -= 500.0 * sig_b

        # Top: r_i - (h - y_i) = r_i - (2 - w - y_i)
        v_t = r - (h_val - y)
        sp_t = eps_b * np.log1p(np.exp(np.clip(v_t / eps_b, -20, 20)))
        obj += 500.0 * np.sum(sp_t)
        sig_t = _sigmoid(v_t / eps_b)
        grad[1 + 2 * N:1 + 3 * N] += 500.0 * sig_t
        grad[0] += 500.0 * np.sum(sig_t)
        grad[1 + N:1 + 2 * N] -= 500.0 * sig_t

        # Overlap penalties: for each pair (i,j), violation = r_i + r_j - dist_ij
        for k in range(len(idx_i)):
            i = idx_i[k]
            j = idx_j[k]
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            dist = np.sqrt(dist_sq + 1e-12)
            viol = r[i] + r[j] - dist
            sp_v = eps_o * np.log1p(np.exp(np.clip(viol / eps_o, -20, 20)))
            obj += 1000.0 * sp_v
            sig_v = _sigmoid(viol / eps_o)
            # d/d r_i
            grad[1 + 2 * N + i] += 1000.0 * sig_v
            # d/d r_j
            grad[1 + 2 * N + j] += 1000.0 * sig_v
            # d/d x_i
            grad[1 + i] -= 1000.0 * sig_v * dx / dist
            # d/d x_j
            grad[1 + j] += 1000.0 * sig_v * dx / dist
            # d/d y_i
            grad[1 + N + i] -= 1000.0 * sig_v * dy / dist
            # d/d y_j
            grad[1 + N + j] += 1000.0 * sig_v * dy / dist

        return obj, grad

    bounds = [(0.2, 1.8)] + [(0.0, 2.0)] * (2 * N) + [(0.0, 1.0)] * N

    try:
        res = minimize(objective_and_grad, v0, method='L-BFGS-B',
                       jac=True, bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-10, 'maxfun': 50000})
    except Exception:
        return w, h, xs, ys, rs

    v = res.x
    w_r = v[0]
    h_r = 2.0 - w_r
    xs_r = v[1:1 + N]
    ys_r = v[1 + N:1 + 2 * N]
    rs_r = np.maximum(v[1 + 2 * N:1 + 3 * N], 0.0)

    if w_r > 0.05 and h_r > 0.05:
        return w_r, h_r, xs_r, ys_r, rs_r
    return w, h, xs, ys, rs


def _validate(w, h, xs, ys, rs, tol=1e-5):
    """Check validity: boundary and non-overlap."""
    if w <= 0 or h <= 0:
        return False
    # Boundary
    if np.any(xs - rs < -tol) or np.any(w - xs - rs < -tol):
        return False
    if np.any(ys - rs < -tol) or np.any(h - ys - rs < -tol):
        return False
    # Overlap
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    return True


def _make_hex_configs():
    """Generate several hexagonal row configurations for 21 circles."""
    configs = [
        [4, 5, 4, 4, 4],   # 5 rows
        [3, 4, 5, 4, 3, 2], # 6 rows
        [4, 4, 4, 4, 4, 1], # 6 rows
        [5, 4, 5, 4, 3],   # 5 rows
        [3, 4, 4, 4, 4, 2], # 6 rows
        [4, 5, 4, 4, 4],   # 5 rows (repeat for variation)
        [5, 4, 4, 4, 4],   # 5 rows
        [2, 4, 5, 4, 4, 2], # 6 rows
        [4, 4, 5, 4, 4],   # 5 rows
        [3, 5, 4, 4, 3, 2], # 6 rows
    ]
    return configs


def _make_grid_configs():
    """Generate several grid configurations for 21 circles."""
    configs = [
        (3, 7),  # 3 rows x 7 cols
        (7, 3),
        (4, 5),  # 4x5 = 20, need 21
        (5, 4),
        (3, 6),  # 18, need 3 more
        (4, 6),  # 24, use 21
        (6, 4),
        (2, 10),
        (10, 2),
        (3, 5),  # 15
    ]
    return configs


def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin before 150s kill

    best_val = -1e10
    best_result = None

    def try_update(w, h, xs, ys, rs):
        nonlocal best_val, best_result
        if not _validate(w, h, xs, ys, rs):
            return
        val = np.sum(rs)
        if val > best_val:
            best_val = val
            best_result = (w, h, xs.copy(), ys.copy(), rs.copy())

    # --- Phase 1: Try various initial configurations ---
    # Hexagonal configs
    hex_configs = _make_hex_configs()
    for cfg in hex_configs:
        if time.time() > deadline - 30.0:
            break
        for aspect in [0.8, 1.0, 1.2, 1.5, 0.7, 1.3, 0.9, 1.1]:
            if time.time() > deadline - 25.0:
                break
            w = 2.0 / (1.0 + aspect)
            h = 2.0 * aspect / (1.0 + aspect)
            xs, ys, w_f, h_f = _make_hex(cfg, w, h)
            # Recompute with actual w, h
            xs, ys, w_f, h_f = _make_hex(cfg, w, h)
            rs = _solve_lp_radii(xs, ys, w_f, h_f)
            try_update(w_f, h_f, xs, ys, rs)

    # Grid configs
    grid_configs = _make_grid_configs()
    for rows, cols in grid_configs:
        if time.time() > deadline - 25.0:
            break
        for aspect in [0.8, 1.0, 1.2, 1.5, 0.7, 1.3]:
            if time.time() > deadline - 20.0:
                break
            w = 2.0 / (1.0 + aspect)
            h = 2.0 * aspect / (1.0 + aspect)
            xs, ys, w_f, h_f = _make_grid(rows, cols, w, h)
            rs = _solve_lp_radii(xs, ys, w_f, h_f)
            try_update(w_f, h_f, xs, ys, rs)

    # Random configs
    for seed in range(20):
        if time.time() > deadline - 20.0:
            break
        for aspect in [0.8, 1.0, 1.2, 1.5]:
            if time.time() > deadline - 15.0:
                break
            w = 2.0 / (1.0 + aspect)
            h = 2.0 * aspect / (1.0 + aspect)
            xs, ys, w_f, h_f = _make_random(w, h, seed * 100 + int(aspect * 10))
            rs = _solve_lp_radii(xs, ys, w_f, h_f)
            try_update(w_f, h_f, xs, ys, rs)

    # --- Phase 2: Refine best candidates with SLSQP and gradient polish ---
    if best_result is not None:
        w, h, xs, ys, rs = best_result

        # Try equal-radius SLSQP refinement
        r_eq = np.mean(rs)
        w2, h2, r2, xs2, ys2 = _refine_slsqp(w, h, xs, ys, r_eq, deadline)
        rs2 = np.full(N, r2)
        try_update(w2, h2, xs2, ys2, rs2)

        # Try gradient polish
        w3, h3, xs3, ys3, rs3 = _polish_gradient(w, h, xs, ys, rs, deadline, max_iter=300)
        try_update(w3, h3, xs3, ys3, rs3)

    # --- Phase 3: More targeted search ---
    # Try to improve by varying aspect ratio around best
    if best_result is not None:
        w0, h0, xs0, ys0, rs0 = best_result
        for aspect in np.linspace(0.5, 2.0, 16):
            if time.time() > deadline - 10.0:
                break
            w = 2.0 / (1.0 + aspect)
            h = 2.0 * aspect / (1.0 + aspect)
            # Scale positions
            xs_new = xs0 * (w / w0)
            ys_new = ys0 * (h / h0)
            rs_new = _solve_lp_radii(xs_new, ys_new, w, h)
            try_update(w, h, xs_new, ys_new, rs_new)

            # Gradient polish on this
            w3, h3, xs3, ys3, rs3 = _polish_gradient(w, h, xs_new, ys_new, rs_new, deadline, max_iter=200)
            try_update(w3, h3, xs3, ys3, rs3)

    # --- Phase 4: Final polish on best ---
    if best_result is not None:
        w, h, xs, ys, rs = best_result
        w4, h4, xs4, ys4, rs4 = _polish_gradient(w, h, xs, ys, rs, deadline, max_iter=500)
        try_update(w4, h4, xs4, ys4, rs4)

    # --- Return best valid result ---
    if best_result is not None:
        w, h, xs, ys, rs = best_result
        return np.column_stack([xs, ys, rs])

    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(N):
        row = i // side
        col = i % side
        x = (col + 0.5) / side
        y = (row + 0.5) / side
        r = 0.5 / side - 1e-3
        cells.append((x, y, r))
    return np.array(cells[:N])
