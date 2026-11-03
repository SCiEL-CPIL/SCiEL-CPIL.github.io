"""Pack 50 equal circles in a unit square, maximize common radius."""
N = 50
DEADLINE = 118.0  # safety margin before the 150s kill

import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import Delaunay


def _radius_fast(c):
    """Exact packing radius in double precision."""
    c = np.asarray(c, dtype=float)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    if np.any(c < -0.5) or np.any(c > 0.5):
        return 0.0
    # wall distances
    w = np.min(np.stack([
        0.5 - c[:, 0], c[:, 0] + 0.5,
        0.5 - c[:, 1], c[:, 1] + 0.5,
    ]), axis=0)
    r = float(np.min(w))
    # pairwise distances
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    r = min(r, float(np.min(d)) * 0.5)
    return r


def _soft_min_radius(c, k=50.0):
    """Smooth surrogate for the min radius using log-sum-exp."""
    pts = np.asarray(c, dtype=float)
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dists, np.inf)
    dw = np.stack([
        0.5 - pts[:, 0], 0.5 + pts[:, 0],
        0.5 - pts[:, 1], 0.5 + pts[:, 1],
    ])
    all_d = np.concatenate([dists.ravel(), dw.ravel()])
    all_d = np.clip(all_d, 1e-10, 1.0)
    m = np.min(all_d)
    val = m - np.log(np.exp(-k * (all_d - m)).sum()) / k
    return val


def _soft_min_radius_grad(c, k=50.0):
    """Gradient of the soft-min radius surrogate."""
    pts = np.asarray(c, dtype=float)
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dists, np.inf)
    dw = np.stack([
        0.5 - pts[:, 0], 0.5 + pts[:, 0],
        0.5 - pts[:, 1], 0.5 + pts[:, 1],
    ])
    all_d = np.concatenate([dists.ravel(), dw.ravel()])
    all_d = np.clip(all_d, 1e-10, 1.0)
    m = np.min(all_d)
    exp_term = np.exp(-k * (all_d - m))
    sw = exp_term.sum()
    weights = exp_term / sw

    grad_pts = np.zeros_like(pts)
    wp = weights[:N * N].reshape(N, N)
    mask = ~np.eye(N, dtype=bool)
    safe = np.where(mask, dists, 1.0)
    factors = np.where(mask, wp / safe, 0.0)
    s = factors.sum(axis=1)
    wpts = (factors[..., None] * pts[None, :, :]).sum(axis=1)
    grad_pts += pts * s[:, None] - wpts

    ww = weights[N * N:].reshape(4, N)
    grad_pts[:, 0] += -ww[0] + ww[1]
    grad_pts[:, 1] += -ww[2] + ww[3]
    return grad_pts


def _delaunay_sparse_objective(c, k=30.0):
    """
    Use Delaunay triangulation to identify critical (close) pairs and
    optimize a weighted sum over those pairs plus walls.
    """
    pts = np.asarray(c, dtype=float)
    try:
        tri = Delaunay(pts)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a, b), max(a, b)))
    except Exception:
        edges = set()

    # Also add wall constraints
    # Build a sparse list of (index, type) where type 0=pair, 1=wall
    # For simplicity, use all Delaunay edges + all walls
    obj = 0.0
    grad = np.zeros_like(pts)
    total_w = 0.0

    for (i, j) in edges:
        d = np.sqrt(np.sum((pts[i] - pts[j]) ** 2))
        d = max(d, 1e-10)
        w = np.exp(-k * d)
        obj += w * (-d)  # minimize -d weighted
        total_w += w
        diff = pts[i] - pts[j]
        grad[i] += w * diff / d
        grad[j] -= w * diff / d

    # Wall contributions
    for i in range(N):
        for axis in range(2):
            for sign in [1, -1]:
                if sign > 0:
                    d = 0.5 - pts[i, axis]
                else:
                    d = pts[i, axis] + 0.5
                d = max(d, 1e-10)
                w = np.exp(-k * d)
                obj += w * (-d)
                total_w += w
                if sign > 0:
                    grad[i, axis] += -w
                else:
                    grad[i, axis] += w

    if total_w > 0:
        obj /= total_w
        grad /= total_w

    return -obj, -grad  # maximize radius -> minimize negative


def _relax_hard_disk(c, r, time_budget=3.0, max_iter=15000):
    """Push overlapping circles apart and keep them inside the square."""
    c = np.array(c, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r, 0.5 - r)
    start = time.time()
    dt = 0.04
    for _ in range(max_iter):
        if time.time() - start > time_budget:
            break
        disp = np.zeros_like(c)
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        mask = d < 2 * r
        if np.any(mask):
            safe = np.where(d > 1e-12, d, 1e-12)
            vec = (c[:, None, :] - c[None, :, :]) / safe[:, :, None]
            overlap = (2 * r - d) / (2 * r)
            overlap = np.where(mask, overlap, 0.0)
            force = overlap[:, :, None] * vec
            disp += force.sum(axis=1)
        # wall push
        for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1)]:
            if sign > 0:
                pen = c[:, axis] - (0.5 - r)
                pen = np.clip(pen, 0, None)
            else:
                pen = (-0.5 + r) - c[:, axis]
                pen = np.clip(pen, 0, None)
            disp[:, axis] += sign * pen / (2 * r)
        c = np.clip(c + dt * disp, -0.5 + r, 0.5 - r)
    return c


def _voronoi_style_refine(c, r, time_budget=3.0, max_iter=5000):
    """
    Voronoi-style refinement: for each circle, check if it's constrained
    by neighbors/walls and nudge it to equalize the minimum distances.
    """
    c = np.array(c, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r, 0.5 - r)
    start = time.time()
    for _ in range(max_iter):
        if time.time() - start > time_budget:
            break
        improved = False
        for i in range(N):
            # Find closest neighbor and closest wall
            d = np.sqrt(((c[i] - c) ** 2).sum(-1))
            d[i] = np.inf
            j = np.argmin(d)
            d_j = d[j]

            # Wall distances for point i
            w = np.min([0.5 - c[i, 0], c[i, 0] + 0.5, 0.5 - c[i, 1], c[i, 1] + 0.5])

            # If the closest constraint is a neighbor, push away
            if d_j < 2 * r + 1e-6:
                diff = c[i] - c[j]
                norm = np.sqrt(np.sum(diff ** 2))
                if norm > 1e-12:
                    push = (2 * r - d_j) * 0.5 * diff / norm
                    c[i] += push
                    improved = True

            # Wall constraints
            for axis in range(2):
                if c[i, axis] < -0.5 + r:
                    c[i, axis] = -0.5 + r
                    improved = True
                if c[i, axis] > 0.5 - r:
                    c[i, axis] = 0.5 - r
                    improved = True

        if not improved:
            break
    return c


def _slsqp_epigraph_polish(c0, r_init, time_budget=5.0, maxiter=200):
    """
    SLSQP with epigraph formulation: maximize t subject to
      t <= |c_i - c_j| / 2  for all i < j
      t <= 0.5 - |c_ik|     for all i, k
    """
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    r_init = max(r_init, 0.01)

    def neg_t(x):
        return -x[2 * N]

    def neg_t_grad(x):
        g = np.zeros(2 * N + 1)
        g[2 * N] = -1.0
        return g

    # Build constraints
    cons = []
    # Pairwise distance constraints: |c_i - c_j| / 2 - t >= 0
    for i in range(N):
        for j in range(i + 1, N):
            def pair_con(x, i=i, j=j):
                ci = x[2 * i:2 * i + 2]
                cj = x[2 * j:2 * j + 2]
                t = x[2 * N]
                d = np.sqrt(np.sum((ci - cj) ** 2))
                return d / 2.0 - t

            def pair_con_grad(x, i=i, j=j):
                g = np.zeros(2 * N + 1)
                ci = x[2 * i:2 * i + 2]
                cj = x[2 * j:2 * j + 2]
                diff = ci - cj
                d = np.sqrt(np.sum(diff ** 2))
                if d > 1e-12:
                    g[2 * i:2 * i + 2] = diff / (2 * d)
                    g[2 * j:2 * j + 2] = -diff / (2 * d)
                g[2 * N] = -1.0
                return g

            cons.append({'type': 'ineq', 'fun': pair_con, 'jac': pair_con_grad})

    # Wall constraints
    for i in range(N):
        for axis in range(2):
            # 0.5 - c[i, axis] - t >= 0
            def wall_con(x, i=i, axis=axis):
                return 0.5 - x[2 * i + axis] - x[2 * N]

            def wall_con_grad(x, i=i, axis=axis):
                g = np.zeros(2 * N + 1)
                g[2 * i + axis] = -1.0
                g[2 * N] = -1.0
                return g

            cons.append({'type': 'ineq', 'fun': wall_con, 'jac': wall_con_grad})

            # c[i, axis] + 0.5 - t >= 0
            def wall_con2(x, i=i, axis=axis):
                return x[2 * i + axis] + 0.5 - x[2 * N]

            def wall_con2_grad(x, i=i, axis=axis):
                g = np.zeros(2 * N + 1)
                g[2 * i + axis] = 1.0
                g[2 * N] = -1.0
                return g

            cons.append({'type': 'ineq', 'fun': wall_con2, 'jac': wall_con2_grad})

    x0 = np.concatenate([c0.ravel(), [r_init]])

    start = time.time()
    try:
        res = minimize(neg_t, x0, jac=neg_t_grad, method='SLSQP',
                       bounds=[(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)],
                       constraints=cons,
                       options={'maxiter': maxiter, 'ftol': 1e-12,
                                'disp': False})
        c = np.clip(res.x[:2 * N].reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0

    return c


def _optimize_lbfgsb(c0, k=50.0, time_budget=4.0, maxiter=200):
    """L-BFGS-B optimization of the soft-min radius surrogate."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)

    def obj_and_grad(x):
        pts = x.reshape(-1, 2)
        val = _soft_min_radius(pts, k)
        grad = _soft_min_radius_grad(pts, k)
        return -val, -grad.ravel()

    try:
        res = minimize(obj_and_grad, c0.ravel(), jac=True, method='L-BFGS-B',
                       bounds=[(-0.5, 0.5)] * (2 * N),
                       options={'maxiter': maxiter, 'ftol': 1e-14,
                                'gtol': 1e-10, 'iprint': 0})
        c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0
    return c


def _init_7x7_plus():
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _init_hexagonal():
    m = 7
    pts = []
    for i in range(m):
        for j in range(m):
            x = (j + 0.5) / m - 0.5
            y = (i + 0.5) / m - 0.5
            if i % 2 == 1:
                x += 0.5 / m
            pts.append((x, y))
    g = np.array(pts)
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _init_jittered_8x8():
    m = 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = []
    for i in range(m):
        for j in range(m):
            pts.append((ticks[j], ticks[i]))
    g = np.array(pts)
    # Remove one corner to get 63, then we need 50, so just take first 50
    return g[:50]


def _init_random(seed=42):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.42, 0.42, (N, 2))


def _grow_radius(c0, time_budget=4.0):
    """Binary search on radius with hard-disk relaxation."""
    r = _radius_fast(c0) * 0.95
    r = max(r, 0.05)
    lo, hi = 0.0, r
    cur = c0
    for _ in range(8):
        if time.time() - start_time > DEADLINE - 15:
            break
        cur = _relax_hard_disk(cur, r, time_budget=2.0, max_iter=5000)
        s = _radius_fast(cur)
        if s >= r - 1e-9:
            lo = r
            r = r * 1.05
            hi = r
        else:
            hi = r
            if hi - lo < 1e-6:
                break
            r = 0.5 * (lo + hi)
    return cur, _radius_fast(cur)


def _final_local_search(c, r, time_budget=3.0):
    """Final local search: try small perturbations to improve radius."""
    c = np.array(c, dtype=float, copy=True)
    best_r = r
    best_c = c.copy()
    start = time.time()
    rng = np.random.default_rng(123)

    for _ in range(200):
        if time.time() - start > time_budget:
            break
        c_trial = c.copy()
        # Random small perturbation
        idx = rng.integers(0, N)
        c_trial[idx] += rng.normal(0, 0.002, 2)
        c_trial = np.clip(c_trial, -0.5 + best_r * 0.9, 0.5 - best_r * 0.9)
        s = _radius_fast(c_trial)
        if s > best_r:
            best_r = s
            best_c = c_trial.copy()

    return best_c, best_r


start_time = time.time()


def construct():
    global start_time
    start_time = time.time()

    best = _init_7x7_plus()
    best_r = _radius_fast(best)

    inits = [_init_7x7_plus(), _init_hexagonal(), _init_jittered_8x8()]
    rng = np.random.default_rng(12345)
    for _ in range(6):
        inits.append(_init_random(rng.integers(0, 100000)))

    # 1) Hard-disk radius expansion from several starts.
    for c0 in inits:
        if time.time() - start_time > DEADLINE - 20:
            break
        cur, s = _grow_radius(c0, time_budget=3.0)
        if s > best_r:
            best_r, best = s, cur

    # 2) Voronoi-style refinement
    for c0 in [best] + inits[:3]:
        if time.time() - start_time > DEADLINE - 15:
            break
        cur = _voronoi_style_refine(c0, best_r, time_budget=2.0, max_iter=3000)
        s = _radius_fast(cur)
        if s > best_r:
            best_r, best = s, cur

    # 3) L-BFGS-B soft-min optimization
    for c0 in [best] + inits[:3]:
        if time.time() - start_time > DEADLINE - 15:
            break
        for k in (40.0, 80.0):
            cur = _optimize_lbfgsb(c0, k=k, time_budget=2.5, maxiter=150)
            s = _radius_fast(cur)
            if s > best_r:
                best_r, best = s, cur

    # 4) SLSQP epigraph polish
    if time.time() - start_time < DEADLINE - 10:
        cur = _slsqp_epigraph_polish(best, best_r, time_budget=6.0, maxiter=200)
        s = _radius_fast(cur)
        if s > best_r:
            best_r, best = s, cur

    # 5) More SLSQP from other starts if time permits
    for c0 in inits[:3]:
        if time.time() - start_time > DEADLINE - 10:
            break
        cur = _slsqp_epigraph_polish(c0, best_r * 0.95, time_budget=4.0, maxiter=150)
        s = _radius_fast(cur)
        if s > best_r:
            best_r, best = s, cur

    # 6) Final local search
    if time.time() - start_time < DEADLINE - 5:
        best, best_r = _final_local_search(best, best_r, time_budget=3.0)

    # 7) Final SLSQP polish at the best radius found
    if time.time() - start_time < DEADLINE - 3:
        cur = _slsqp_epigraph_polish(best, best_r, time_budget=3.0, maxiter=100)
        s = _radius_fast(cur)
        if s > best_r:
            best_r, best = s, cur

    return best
