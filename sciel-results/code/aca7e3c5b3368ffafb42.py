"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Strategy:
 1. Generate diverse initial configurations (hex, grid, random) across many aspect ratios.
 2. For each, run a smooth gradient-based polish (L-BFGS-B) jointly optimizing
    aspect ratio, all centers, and all radii with softplus penalty for violations.
 3. Post-process: project radii to a feasible LP solution given final centers,
    then scale to perimeter <= 4.
 4. Track best by sum of radii; return best valid result.
"""
import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay

N = 21


def _softplus(x, k=10.0):
    """Numerically stable softplus: (1/k)*log(1+exp(k*x))."""
    return np.where(x > 20.0 / k, x, (1.0 / k) * np.log1p(np.exp(k * x)))


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
    Solve LP for per-circle radii given fixed centers and rectangle [0,w]x[0,h].
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


def _make_hex(rows):
    """Hexagonal layout of given per-row counts, normalized to unit span."""
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


def _make_grid(rows, cols):
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
    xs = rng.uniform(0.15, 0.85, N) * w
    ys = rng.uniform(0.15, 0.85, N) * h
    return xs, ys, w, h


def _polish(w, h, xs, ys, rs, deadline, max_iter=400):
    """
    Smooth gradient-based polish using L-BFGS-B with softplus penalty.
    Variables: [w, x0..x20, y0..y20, r0..r20].  h = 2 - w.
    Maximizes sum(r_i) - penalty(violations).
    Returns (w, h, xs, ys, rs) or input on failure.
    """
    if time.time() > deadline - 5.0:
        return w, h, xs, ys, rs

    # State: [w, x0..x20, y0..y20, r0..r20]
    n_vars = 1 + N + N + N  # 64
    v0 = np.zeros(n_vars)
    v0[0] = w
    v0[1:1+N] = xs
    v0[1+N:1+2*N] = ys
    v0[1+2*N:1+3*N] = np.maximum(rs, 0.01)

    k = 15.0  # softplus sharpness

    def objective_and_grad(v):
        w_v = v[0]
        h_v = 2.0 - w_v
        xs_v = v[1:1+N]
        ys_v = v[1+N:1+2*N]
        rs_v = v[1+2*N:1+3*N]

        if w_v <= 0.15 or h_v <= 0.15:
            return 1e6, np.zeros(n_vars)

        # Boundary violations
        # x - r >= 0, w - x - r >= 0, y - r >= 0, h - y - r >= 0
        viol_left = -(xs_v - rs_v)
        viol_right = -(w_v - xs_v - rs_v)
        viol_bottom = -(ys_v - rs_v)
        viol_top = -(h_v - ys_v - rs_v)

        # Pairwise overlap: r_i + r_j - d_ij
        pairwise_viol = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((xs_v[i]-xs_v[j])**2 + (ys_v[i]-ys_v[j])**2)
                pv = rs_v[i] + rs_v[j] - d
                pairwise_viol[i, j] = pv
                pairwise_viol[j, i] = pv

        all_viol = np.concatenate([viol_left, viol_right, viol_bottom, viol_top,
                                   pairwise_viol.ravel()])
        penalty = np.sum(_softplus(all_viol, k))
        obj = -np.sum(rs_v) + penalty

        # Gradient
        grad = np.zeros(n_vars)

        # d/dw from boundary right and top
        # viol_right_i = -(w - x_i - r_i) => d/dw = -1
        # viol_top_i = -(h - y_i - r_i) = -(2-w-y_i-r_i) => d/dw = +1
        # pairwise doesn't depend on w directly
        grad[0] = 0.0
        # d obj/dw = d(-sum r)/dw + d(penalty)/dw
        # d(penalty)/dw = sum_i [ sigma(k*(-(w-x_i-r_i)))*(-k)*(-1) + sigma(k*(-(2-w-y_i-r_i)))*(-k)*(1) ]
        # where sigma(x) = 1/(1+exp(-x))
        for i in range(N):
            gr = -k / (1.0 + np.exp(-k * (w_v - xs_v[i] - rs_v[i])))
            gt = -k / (1.0 + np.exp(-k * (h_v - ys_v[i] - rs_v[i])))
            # d viol_right/dw = -1, so d softplus/dw = sigma * k * (-1) ... wait
            # softplus(viol) = (1/k)log(1+exp(k*viol))
            # d/dw softplus = exp(k*viol)/(1+exp(k*viol)) * k * d(viol)/dw = sigmoid(k*viol)*k*d(viol)/dw
            # viol_right = -(w - x - r), d/dw = -1
            # so contribution = sigmoid(k*viol_right)*k*(-1)
            # viol_top = -(h - y - r) = w + y + r - 2, d/dw = +1
            # contribution = sigmoid(k*viol_top)*k*(1)
            sig_r = 1.0 / (1.0 + np.exp(-k * (-(w_v - xs_v[i] - rs_v[i]))))
            sig_t = 1.0 / (1.0 + np.exp(-k * (-(h_v - ys_v[i] - rs_v[i]))))
            grad[0] += sig_r * k * (-1.0) + sig_t * k * (1.0)

        # d/dx_i
        for i in range(N):
            # boundary left: viol = -(x_i - r_i), d/dx_i = -1
            sig_l = 1.0 / (1.0 + np.exp(-k * (-(xs_v[i] - rs_v[i]))))
            # boundary right: viol = -(w - x_i - r_i), d/dx_i = +1
            sig_r = 1.0 / (1.0 + np.exp(-k * (-(w_v - xs_v[i] - rs_v[i]))))
            grad[1+i] = -1.0 + sig_l * k * (-1.0) + sig_r * k * (1.0)

            # pairwise
            for j in range(N):
                if j == i:
                    continue
                d = np.sqrt((xs_v[i]-xs_v[j])**2 + (ys_v[i]-ys_v[j])**2)
                if d < 1e-12:
                    continue
                pv = rs_v[i] + rs_v[j] - d
                sig_p = 1.0 / (1.0 + np.exp(-k * pv))
                # d pv/dx_i = -(x_i - x_j)/d
                grad[1+i] += sig_p * k * (-(xs_v[i] - xs_v[j]) / d)

        # d/dy_i
        for i in range(N):
            sig_b = 1.0 / (1.0 + np.exp(-k * (-(ys_v[i] - rs_v[i]))))
            sig_t = 1.0 / (1.0 + np.exp(-k * (-(h_v - ys_v[i] - rs_v[i]))))
            grad[1+N+i] = -1.0 + sig_b * k * (-1.0) + sig_t * k * (1.0)
            for j in range(N):
                if j == i:
                    continue
                d = np.sqrt((xs_v[i]-xs_v[j])**2 + (ys_v[i]-ys_v[j])**2)
                if d < 1e-12:
                    continue
                pv = rs_v[i] + rs_v[j] - d
                sig_p = 1.0 / (1.0 + np.exp(-k * pv))
                grad[1+N+i] += sig_p * k * (-(ys_v[i] - ys_v[j]) / d)

        # d/dr_i
        for i in range(N):
            # boundary: viol_left d/dr_i = -1, viol_right d/dr_i = -1, etc.
            sig_l = 1.0 / (1.0 + np.exp(-k * (-(xs_v[i] - rs_v[i]))))
            sig_r = 1.0 / (1.0 + np.exp(-k * (-(w_v - xs_v[i] - rs_v[i]))))
            sig_b = 1.0 / (1.0 + np.exp(-k * (-(ys_v[i] - rs_v[i]))))
            sig_t = 1.0 / (1.0 + np.exp(-k * (-(h_v - ys_v[i] - rs_v[i]))))
            grad[1+2*N+i] = -1.0 + sig_l*k*(-1.0) + sig_r*k*(-1.0) + sig_b*k*(-1.0) + sig_t*k*(-1.0)
            for j in range(N):
                if j == i:
                    continue
                d = np.sqrt((xs_v[i]-xs_v[j])**2 + (ys_v[i]-ys_v[j])**2)
                if d < 1e-12:
                    continue
                pv = rs_v[i] + rs_v[j] - d
                sig_p = 1.0 / (1.0 + np.exp(-k * pv))
                grad[1+2*N+i] += sig_p * k * 1.0

        return obj, grad

    bounds = [(0.15, 1.85)] + [(0.0, 2.0)] * (N + N) + [(0.001, 1.0)] * N

    try:
        res = minimize(objective_and_grad, v0, method='L-BFGS-B',
                       jac=True, bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8,
                                'maxfun': 8000, 'disp': False})
    except Exception:
        return w, h, xs, ys, rs

    if res.success or res.fun < 1e5:
        w_r = res.x[0]
        h_r = 2.0 - w_r
        xs_r = res.x[1:1+N]
        ys_r = res.x[1+N:1+2*N]
        rs_r = np.maximum(res.x[1+2*N:1+3*N], 0.0)
        if w_r > 0.1 and h_r > 0.1:
            return w_r, h_r, xs_r, ys_r, rs_r
    return w, h, xs, ys, rs


def _make_result(w, h, xs, ys):
    """Build a valid (N,3) result using LP radii, then scale to perimeter <= 4."""
    rs = _solve_lp_radii(xs, ys, w, h)
    result = np.zeros((N, 3))
    result[:, 0] = xs
    result[:, 1] = ys
    result[:, 2] = np.maximum(rs, 0.0)
    _scale_to_perimeter(result)
    return result


def _scale_to_perimeter(result):
    """Scale result about its bounding-box center so perimeter <= 4."""
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
        for i in range(N):
            result[i, 0] = cx + (result[i, 0] - cx) * scale
            result[i, 1] = cy + (result[i, 1] - cy) * scale
            result[i, 2] = result[i, 2] * scale
    return result


def _validity_margin(result, tol=1e-5):
    """Return minimum clearance (positive = valid)."""
    xs = result[:, 0]
    ys = result[:, 1]
    rs = result[:, 2]
    if np.any(rs < 1e-6):
        return -1.0
    m = float('inf')
    # boundary
    m = min(m, float(np.min(xs - rs)))
    m = min(m, float(np.min(ys - rs)))
    # For perimeter constraint, we need the bounding box to fit in some w x h with w+h=2
    # After scaling, check that circles are within [0, w] x [0, h] for w+h=2
    # Actually after _scale_to_perimeter, the bounding box has perimeter exactly 4 (or less)
    # Let's check: min of (x-r) should be >= 0 after translating to origin
    # The scaling centers on bbox center, so we need to verify
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    bw = max_x - min_x
    bh = max_y - min_y
    if 2.0 * (bw + bh) > 4.0 + 1e-5:
        return -1.0
    # Check pairwise
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
            m = min(m, d - (rs[i] + rs[j]))
    return m


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    np.random.seed(42)

    best_result = None
    best_value = -np.inf

    def consider(result):
        nonlocal best_result, best_value
        if result is None:
            return
        if _validity_margin(result) < -1e-5:
            return
        val = float(np.sum(result[:, 2]))
        if val > best_value:
            best_value = val
            best_result = result.copy()

    # ---------- Generate initial configurations ----------
    inits = []

    # Hexagonal layouts
    hex_rows_list = [
        [7, 7, 7], [4, 5, 4, 4, 4], [4, 4, 4, 4, 4, 1],
        [3, 4, 4, 4, 4, 2], [5, 5, 5, 3, 3], [6, 7, 8],
        [7, 6, 8], [8, 7, 6], [5, 6, 5, 5], [6, 5, 6, 4],
    ]
    for rows in hex_rows_list:
        if sum(rows) < N:
            continue
        xs, ys, w, h = _make_hex(rows)
        inits.append((w, h, xs, ys))

    # Grid layouts
    for rows, cols in [(5, 5), (3, 7), (4, 6), (7, 3), (6, 4), (7, 4), (4, 5)]:
        xs, ys, w, h = _make_grid(rows, cols)
        inits.append((w, h, xs, ys))

    # Random inits across aspect ratios
    aspect_ratios = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    for ar in aspect_ratios:
        w = ar / (1.0 + ar) * 2.0
        h = 2.0 - w
        for seed in [100 + int(ar * 10)]:
            xs, ys, w, h = _make_random(w, h, seed)
            inits.append((w, h, xs, ys))

    # ---------- Process each init ----------
    for idx, (w, h, xs, ys) in enumerate(inits):
        if time.time() > deadline - 10.0:
            break

        # Quick LP result
        res_lp = _make_result(w, h, xs, ys)
        consider(res_lp)

        # Get initial radii from LP for polish
        rs0 = _solve_lp_radii(xs, ys, w, h)

        # Gradient polish
        if time.time() < deadline - 8.0:
            w2, h2, xs2, ys2, rs2 = _polish(w, h, xs, ys, rs0, deadline)
            res_polished = _make_result(w2, h2, xs2, ys2)
            consider(res_polished)

            # Second polish with perturbed start
            if time.time() < deadline - 5.0:
                rng = np.random.RandomState(idx + 200)
                xs_p = xs2 + rng.normal(0, 0.01, N)
                ys_p = ys2 + rng.normal(0, 0.01, N)
                xs_p = np.clip(xs_p, 0.01, w2 - 0.01)
                ys_p = np.clip(ys_p, 0.01, h2 - 0.01)
                w3, h3, xs3, ys3, rs3 = _polish(w2, h2, xs_p, ys_p, rs0, deadline, max_iter=200)
                res_perturbed = _make_result(w3, h3, xs3, ys3)
                consider(res_perturbed)

    # ---------- Final fallback ----------
    if best_result is None or best_value <= 0:
        w = 1.0
        h = 1.0
        side = 5
        result = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            result[i, 0] = (col + 0.5) / side * w
            result[i, 1] = (row + 0.5) / side * h
            result[i, 2] = 0.48 / side
        _scale_to_perimeter(result)
        return result

    return best_result
