import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay

N = 21

def _softplus(x, beta=100.0):
    """Numerically stable softplus: log(1+exp(beta*x))/beta"""
    x_beta = beta * x
    # Use log1p(exp(-x_beta)) for x_beta > 0 to avoid overflow
    # Use log1p(exp(x_beta)) for x_beta <= 0
    result = np.where(x_beta > 0, 
                     x + np.log1p(np.exp(-x_beta)) / beta,
                     np.log1p(np.exp(x_beta)) / beta)
    return result

def _softplus_grad(x, beta=100.0):
    """Gradient of softplus: sigmoid(beta*x)"""
    # Stable sigmoid
    return np.where(x_beta > 0, 
                    1.0 / (1.0 + np.exp(-x_beta)),
                    np.exp(x_beta) / (1.0 + np.exp(x_beta)))
    # Note: x_beta is beta * x. Need to be careful with scope.
    # Let's implement a standalone stable sigmoid for beta*x
    zx = beta * x
    return np.where(zx > 0, 
                    1.0 / (1.0 + np.exp(-zx)),
                    np.exp(zx) / (1.0 + np.exp(zx)))

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
    Maximize sum(r) s.t. r_i + r_j <= d_ij (Delaunay edges), r_i <= boundary, r_i >= 0.
    Returns (radii, duals) where duals is a list of (kind, i, j, value).
    """
    limits = np.minimum(np.minimum(xs, w - xs), np.minimum(ys, h - ys))
    if np.any(limits < 0):
        return np.zeros(N), []

    A_list = []
    b_list = []
    edges = _delaunay_edges(xs, ys)
    pair_idx = []
    if edges is not None:
        for i, j in edges:
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            A_list.append(row)
            b_list.append(dist)
            pair_idx.append((i, j))

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
            radii = np.maximum(0.0, res.x)
            duals = []
            # Extract duals if available
            if hasattr(res, 'ineqlin') and res.ineqlin is not None and hasattr(res.ineqlin, 'marginals'):
                marg = res.ineqlin.marginals
                for k, (i, j) in enumerate(pair_idx):
                    if k < len(marg):
                        duals.append(('pair', i, j, abs(marg[k])))
                offset = len(pair_idx)
                for i in range(N):
                    if offset + i < len(marg):
                        duals.append(('bound', i, -1, abs(marg[offset + i])))
            return radii, duals
    except Exception:
        pass
    return np.maximum(0.0, limits * 0.9), []

def _max_equal_radius(xs, ys, w, h):
    """Largest r such that equal-radius packing is feasible."""
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

def _make_hex(rows, aspect_ratio=1.0):
    """
    Hexagonal layout with given per-row counts.
    Returns xs, ys, w, h.
    """
    pts = []
    for k, n in enumerate(rows):
        y = k * np.sqrt(3.0)
        off = 1.0 if k % 2 == 1 else 0.0
        for i in range(n):
            pts.append((i * 2.0 + off, y))
    pts = np.array(pts[:N])
    if len(pts) < N:
        return None
    xs = pts[:, 0]
    ys = pts[:, 1]
    cx = 0.5 * (xs.max() + xs.min())
    cy = 0.5 * (ys.max() + ys.min())
    xs = xs - cx
    ys = ys - cy
    wc = xs.max() - xs.min()
    hc = ys.max() - ys.min()
    
    # Apply aspect ratio by scaling x or y? 
    # Perimeter constraint: 2(w+h)=4 => w+h=2.
    # We want to fit the points into a rectangle of aspect ratio AR = w/h.
    # Let w = AR * h. Then AR*h + h = 2 => h = 2/(AR+1), w = 2*AR/(AR+1).
    
    # We need to scale the points to fit within w x h.
    # The points currently span wc x hc.
    # We scale x by sx and y by sy such that wc*sx <= w and hc*sy <= h.
    # To maintain hex structure, we might want sx=sy, but if aspect ratio is forced, we shear or stretch.
    # For now, simple scaling to fit the box.
    sx = w / wc if wc > 0 else 1.0
    sy = h / hc if hc > 0 else 1.0
    s = min(sx, sy)
    
    xs_f = xs * s + 0.5 * w
    ys_f = ys * s + 0.5 * h
    return xs_f, ys_f, w, h

def _make_grid(rows, cols, aspect_ratio=1.0):
    pts = []
    for j in range(rows):
        for i in range(cols):
            if len(pts) >= N:
                break
            pts.append((i * 2.0, j * 2.0))
    pts = np.array(pts[:N])
    if len(pts) < N:
        return None
    xs = pts[:, 0]
    ys = pts[:, 1]
    cx = 0.5 * (xs.max() + xs.min())
    cy = 0.5 * (ys.max() + ys.min())
    xs = xs - cx
    ys = ys - cy
    wt = (xs.max() - xs.min()) + 2.0
    ht = (ys.max() - ys.min()) + 2.0
    
    w = 2.0 * aspect_ratio / (1.0 + aspect_ratio)
    h = 2.0 / (1.0 + aspect_ratio)
    
    sx = w / wt if wt > 0 else 1.0
    sy = h / ht if ht > 0 else 1.0
    s = min(sx, sy)
    
    xs_f = xs * s + 0.5 * w
    ys_f = ys * s + 0.5 * h
    return xs_f, ys_f, w, h

def _make_random(w, h, seed):
    rng = np.random.RandomState(seed)
    xs = rng.uniform(0.1, 0.9, N) * w
    ys = rng.uniform(0.1, 0.9, N) * h
    return xs, ys, w, h

def _make_farthest_point(w, h, seed):
    """Farthest-point sampling initialization for better spread."""
    rng = np.random.RandomState(seed)
    xs = np.zeros(N)
    ys = np.zeros(N)
    xs[0] = rng.uniform(0.2, 0.8) * w
    ys[0] = rng.uniform(0.2, 0.8) * h
    for k in range(1, N):
        cand_x = rng.uniform(0.05, 0.95, 200) * w
        cand_y = rng.uniform(0.05, 0.95, 200) * h
        min_dists = np.full(200, np.inf)
        for j in range(k):
            d = np.sqrt((cand_x - xs[j]) ** 2 + (cand_y - ys[j]) ** 2)
            min_dists = np.minimum(min_dists, d)
        bd = np.minimum(np.minimum(cand_x, w - cand_x), np.minimum(cand_y, h - cand_y))
        min_dists = np.minimum(min_dists, bd)
        best = np.argmax(min_dists)
        xs[k] = cand_x[best]
        ys[k] = cand_y[best]
    return xs, ys, w, h

def _dual_driven_optimize(xs, ys, w, h, max_iters=200, deadline=None):
    """
    LP-dual driven center optimization.
    """
    xs = xs.copy().astype(float)
    ys = ys.copy().astype(float)
    best_val = -np.inf
    best_xs, best_ys, best_rs = xs.copy(), ys.copy(), np.zeros(N)

    step = 0.02
    stall = 0

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 5.0:
            break

        radii, duals = _solve_lp_radii(xs, ys, w, h)
        val = float(np.sum(radii))

        if val > best_val:
            best_val = val
            best_xs = xs.copy()
            best_ys = ys.copy()
            best_rs = radii.copy()
            stall = 0
        else:
            stall += 1

        if stall > 20:
            step *= 0.7
            xs += np.random.normal(0, 0.005, N)
            ys += np.random.normal(0, 0.005, N)
            xs = np.clip(xs, 1e-4, w - 1e-4)
            ys = np.clip(ys, 1e-4, h - 1e-4)
            stall = 0
            continue

        # Compute forces from duals
        fx = np.zeros(N)
        fy = np.zeros(N)
        for kind, i, j, dv in duals:
            if dv < 1e-8:
                continue
            if kind == 'pair':
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < 1e-10:
                    continue
                fx[i] += dv * dx / d
                fy[i] += dv * dy / d
                fx[j] -= dv * dx / d
                fy[j] -= dv * dy / d
            elif kind == 'bound':
                # Push away from boundaries
                # Determine which boundary is tight
                # Left: x < r
                if xs[i] <= radii[i] + 1e-6:
                    fx[i] += dv
                # Right: w - x < r
                if w - xs[i] <= radii[i] + 1e-6:
                    fx[i] -= dv
                # Bottom: y < r
                if ys[i] <= radii[i] + 1e-6:
                    fy[i] += dv
                # Top: h - y < r
                if h - ys[i] <= radii[i] + 1e-6:
                    fy[i] -= dv

        fmag = np.sqrt(fx * fx + fy * fy)
        fmag = np.maximum(fmag, 1e-10)
        fx = fx / fmag
        fy = fy / fmag

        xs += step * fx
        ys += step * fy
        xs = np.clip(xs, 1e-4, w - 1e-4)
        ys = np.clip(ys, 1e-4, h - 1e-4)

    return best_xs, best_ys, best_rs, best_val

def _vectorized_polish(w, h, xs, ys, rs, deadline, max_iter=1000):
    """
    High-performance vectorized L-BFGS-B polish with softplus penalties.
    Variables: [w, x0..x20, y0..y20, r0..r20]. h = 2 - w.
    """
    if time.time() > deadline - 5.0:
        return w, h, xs, ys, rs

    v0 = np.zeros(1 + 3 * N)
    v0[0] = w
    v0[1:1 + N] = xs
    v0[1 + N:1 + 2 * N] = ys
    v0[1 + 2 * N:1 + 3 * N] = rs

    # Precompute indices for all pairs
    idx_i, idx_j = np.triu_indices(N, k=1)

    def obj_grad(v):
        w_val = v[0]
        h_val = 2.0 - w_val
        x = v[1:1 + N]
        y = v[1 + N:1 + 2 * N]
        r = v[1 + 2 * N:1 + 3 * N]

        if w_val < 0.01 or h_val < 0.01:
            return 1e10, np.zeros_like(v)

        obj = -np.sum(r)
        grad = np.zeros_like(v)
        grad[1 + 2 * N:1 + 3 * N] = -1.0

        # Beta for softplus
        beta = 500.0

        # --- Boundary Constraints ---
        # r - x >= 0  => x - r <= 0. Violation: r - x.
        v_l = r - x
        sp_l = _softplus(v_l, beta)
        obj += np.sum(sp_l)
        g_l = _softplus_grad(v_l, beta)
        grad[1:1 + N] += -g_l
        grad[1 + 2 * N:1 + 3 * N] += g_l

        # x + r - w <= 0. Violation: x + r - w.
        v_r = x + r - w_val
        sp_r = _softplus(v_r, beta)
        obj += np.sum(sp_r)
        g_r = _softplus_grad(v_r, beta)
        grad[0] += np.sum(-g_r) # dw/d(x+r-w) = -1
        grad[1:1 + N] += g_r
        grad[1 + 2 * N:1 + 3 * N] += g_r

        # r - y >= 0
        v_b = r - y
        sp_b = _softplus(v_b, beta)
        obj += np.sum(sp_b)
        g_b = _softplus_grad(v_b, beta)
        grad[1 + N:1 + 2 * N] += -g_b
        grad[1 + 2 * N:1 + 3 * N] += g_b

        # y + r - h <= 0. h = 2 - w. y + r - (2 - w) = y + r + w - 2.
        v_t = y + r + w_val - 2.0
        sp_t = _softplus(v_t, beta)
        obj += np.sum(sp_t)
        g_t = _softplus_grad(v_t, beta)
        grad[0] += np.sum(g_t) # dw/d(y+r+w-2) = 1
        grad[1 + N:1 + 2 * N] += g_t
        grad[1 + 2 * N:1 + 3 * N] += g_t

        # --- Non-Overlap Constraints ---
        # r_i + r_j - dist <= 0
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(dist_sq + 1e-12)
        min_dist = r[idx_i] + r[idx_j]
        v_o = min_dist - dist
        
        sp_o = _softplus(v_o, beta)
        obj += np.sum(sp_o)
        g_o = _softplus_grad(v_o, beta)
        
        # Gradients for overlap
        # d(v_o)/dx_i = -dx_i / dist
        coeff = g_o / dist
        np.add.at(grad, 1 + idx_i, -coeff * dx)
        np.add.at(grad, 1 + idx_j, coeff * dx)
        np.add.at(grad, 1 + N + idx_i, -coeff * dy)
        np.add.at(grad, 1 + N + idx_j, coeff * dy)
        np.add.at(grad, 1 + 2 * N + idx_i, g_o)
        np.add.at(grad, 1 + 2 * N + idx_j, g_o)

        return obj, grad

    bounds = [(0.05, 1.95)] + [(0.0, 2.0)] * (3 * N)

    try:
        res = minimize(obj_grad, v0, method='L-BFGS-B', jac=True,
                       bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
    except Exception:
        return w, h, xs, ys, rs

    if res.success or res.fun < 1e3:
        w_r = res.x[0]
        h_r = 2.0 - w_r
        xs_r = res.x[1:1 + N]
        ys_r = res.x[1 + N:1 + 2 * N]
        rs_r = np.maximum(0.0, res.x[1 + 2 * N:1 + 3 * N])
        if w_r > 0.01 and h_r > 0.01:
            return w_r, h_r, xs_r, ys_r, rs_r
    return w, h, xs, ys, rs

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
    """Return minimum clearance (positive = valid). Vectorized."""
    xs = result[:, 0]
    ys = result[:, 1]
    rs = result[:, 2]
    if np.any(rs < 1e-6):
        return -1.0
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    bw = max_x - min_x
    bh = max_y - min_y
    perim = 2.0 * (bw + bh)
    if perim > 4.0 + tol:
        return -1.0
    idx_i, idx_j = np.triu_indices(N, k=1)
    dx = xs[idx_i] - xs[idx_j]
    dy = ys[idx_i] - ys[idx_j]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = rs[idx_i] + rs[idx_j]
    m = np.min(dist - min_dist)
    return float(m)

def _make_result(w, h, xs, ys, rs):
    result = np.zeros((N, 3))
    result[:, 0] = xs
    result[:, 1] = ys
    result[:, 2] = np.maximum(rs, 0.0)
    _scale_to_perimeter(result)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

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

    # ---------- Initial Configurations ----------
    inits = []
    
    # Hexagonal layouts
    hex_rows_configs = [
        [7, 7, 7],
        [4, 5, 4, 4, 4],
        [4, 4, 4, 4, 4, 1],
        [3, 4, 4, 4, 4, 2],
        [5, 5, 5, 3, 3],
        [6, 7, 8],
        [7, 7, 7],
        [5, 6, 5, 5],
        [3, 4, 4, 4, 3, 3],
        [4, 4, 4, 4, 3, 2],
    ]
    aspects = [0.8, 0.9, 1.0, 1.1, 1.2, 1.5]
    for rows in hex_rows_configs:
        if sum(rows) < N:
            continue
        for ar in aspects:
            res = _make_hex(rows, ar)
            if res:
                inits.append(res)
                
    # Grid layouts
    grid_configs = [(5, 5), (3, 7), (4, 6), (7, 3), (6, 4)]
    for rows, cols in grid_configs:
        for ar in aspects:
            res = _make_grid(rows, cols, ar)
            if res:
                inits.append(res)
                
    # Random and Farthest Point
    for seed in range(10):
        w = 1.0 + 0.2 * (seed - 4.5) / 9.0
        h = 2.0 - w
        if h > 0.2:
            inits.append(_make_random(w, h, seed + 100))
            
    for seed in range(5):
        w = 1.0 + 0.2 * (seed - 2.0) / 4.0
        h = 2.0 - w
        if h > 0.2:
            inits.append(_make_farthest_point(w, h, seed + 200))

    # ---------- Process Initials ----------
    for idx, (w, h, xs, ys) in enumerate(inits):
        if time.time() > deadline - 25.0:
            break

        # 1. LP Initial Radii
        rs_lp, _ = _solve_lp_radii(xs, ys, w, h)
        res_lp = _make_result(w, h, xs, ys, rs_lp)
        consider(res_lp)

        # 2. Dual Driven Optimization
        if time.time() < deadline - 20.0:
            xs_d, ys_d, rs_d, val_d = _dual_driven_optimize(xs, ys, w, h, max_iters=150, deadline=deadline)
            if val_d > 0:
                res_dual = _make_result(w, h, xs_d, ys_d, rs_d)
                consider(res_dual)
                # 3. Vectorized Polish on Dual Result
                if time.time() < deadline - 10.0:
                    w_p, h_p, xs_p, ys_p, rs_p = _vectorized_polish(w, h, xs_d, ys_d, rs_d, deadline, max_iter=500)
                    res_pol = _make_result(w_p, h_p, xs_p, ys_p, rs_p)
                    consider(res_pol)
            else:
                # Fallback to LP result if dual failed
                rs_lp2, _ = _solve_lp_radii(xs, ys, w, h)
                if time.time() < deadline - 10.0:
                    w_p, h_p, xs_p, ys_p, rs_p = _vectorized_polish(w, h, xs, ys, rs_lp2, deadline, max_iter=500)
                    res_pol = _make_result(w_p, h_p, xs_p, ys_p, rs_p)
                    consider(res_pol)

    # ---------- Final Global Polish ----------
    if best_result is not None and time.time() < deadline - 15.0:
        xs_b = best_result[:, 0].copy()
        ys_b = best_result[:, 1].copy()
        rs_b = best_result[:, 2].copy()
        min_x = np.min(xs_b - rs_b)
        max_x = np.max(xs_b + rs_b)
        min_y = np.min(ys_b - rs_b)
        max_y = np.max(ys_b + rs_b)
        w_b = max_x - min_x
        h_b = max_y - min_y
        xs_b = xs_b - min_x
        ys_b = ys_b - min_y
        
        # High precision polish
        w_f, h_f, xs_f, ys_f, rs_f = _vectorized_polish(w_b, h_b, xs_b, ys_b, rs_b, deadline, max_iter=1000)
        res_final = _make_result(w_f, h_f, xs_f, ys_f, rs_f)
        consider(res_final)
        
        # One more pass with even higher beta if time permits? 
        # The _vectorized_polish uses fixed beta=500. 
        # We could do a second call with higher beta, but L-BFGS-B might struggle.
        # Let's just do one more standard polish to be safe.
        if time.time() < deadline - 5.0:
             w_f2, h_f2, xs_f2, ys_f2, rs_f2 = _vectorized_polish(w_f, h_f, xs_f, ys_f, rs_f, deadline, max_iter=500)
             res_final2 = _make_result(w_f2, h_f2, xs_f2, ys_f2, rs_f2)
             consider(res_final2)

    # ---------- Fallback ----------
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
