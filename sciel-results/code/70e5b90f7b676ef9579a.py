"""Circle packing in a perimeter-4 rectangle: 21 circles, maximize sum of radii.

Combines a fast vectorized SLSQP (single vectorized pair constraint with analytic
Jacobian) with a *free* rectangle aspect ratio (w is a decision variable, h=2-w),
a coarse->fine multi-start search over aspect ratios, and a penalty-based fallback.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
NP = N * (N - 1) // 2
DEADLINE = 145.0  # safety margin below 150s kill

def _pairs_idx():
    ii = np.repeat(np.arange(N), N - 1)
    jj = np.arange(N)[:, None] + np.arange(1, N)[None, :]
    return ii.ravel(), jj.ravel()

PI, PJ = _pairs_idx()

def _objective(x):
    # layout: [x1..xn, y1..yn, r1..rn, w]
    r = x[2 * N:3 * N]
    w = x[-1]
    h = 2.0 - w
    if w <= 0.05 or h <= 0.05:
        return 1e6
    return -np.sum(r)

def _objective_grad(x):
    g = np.zeros(3 * N + 1)
    g[2 * N:3 * N] = -1.0
    return g

def _boundary_fun(x):
    n = N
    xs = x[:n]; ys = x[n:2 * n]; rs = x[2 * n:3 * n]; w = x[-1]; h = 2.0 - w
    return np.concatenate([xs - rs, w - xs - rs, ys - rs, h - ys - rs])

def _boundary_jac(x):
    n = N
    g = np.zeros((4 * n, 3 * n + 1))
    g[0:n, 0:n] = np.eye(n)
    g[0:n, 2 * n:3 * n] = -np.eye(n)
    g[n:2 * n, 0:n] = -np.eye(n)
    g[n:2 * n, 2 * n:3 * n] = -np.eye(n)
    g[2 * n:3 * n, n:2 * n] = np.eye(n)
    g[2 * n:3 * n, 2 * n:3 * n] = -np.eye(n)
    g[3 * n:4 * n, n:2 * n] = -np.eye(n)
    g[3 * n:4 * n, 2 * n:3 * n] = -np.eye(n)
    # d(h - y - r)/dw = d(2 - w - y - r)/dw = -1  (per circle)
    g[3 * n:4 * n, 3 * n] = -1.0
    return g

def _pairs_fun(x):
    xs = x[:N]; ys = x[N:2 * N]; rs = x[2 * N:3 * N]
    dx = xs[PI] - xs[PJ]
    dy = ys[PI] - ys[PJ]
    d2 = dx * dx + dy * dy
    rs_ = rs[PI] + rs[PJ]
    return d2 - rs_ * rs_

def _pairs_jac(x):
    xs = x[:N]; ys = x[N:2 * N]; rs = x[2 * N:3 * N]
    dx = xs[PI] - xs[PJ]
    dy = ys[PI] - ys[PJ]
    rsum = rs[PI] + rs[PJ]
    g = np.zeros((NP, 3 * N + 1))
    g[np.arange(NP), PI] = 2.0 * dx
    g[np.arange(NP), N + PI] = 2.0 * dy
    g[np.arange(NP), 2 * N + PI] = -2.0 * rsum
    g[np.arange(NP), PJ] = -2.0 * dx
    g[np.arange(NP), N + PJ] = -2.0 * dy
    g[np.arange(NP), 2 * N + PJ] = -2.0 * rsum
    return g

def _radius_fun(x):
    return x[2 * N:3 * N] - 1e-4

def _radius_jac(x):
    g = np.zeros((N, 3 * N + 1))
    g[:, 2 * N:3 * N] = np.eye(N)
    return g

def _cons():
    return [
        {'type': 'ineq', 'fun': _boundary_fun, 'jac': _boundary_jac},
        {'type': 'ineq', 'fun': _pairs_fun, 'jac': _pairs_jac},
        {'type': 'ineq', 'fun': _radius_fun, 'jac': _radius_jac},
    ]

def _grid_seed(w, h, n_cols, n_rows):
    xs = (np.arange(n_cols) + 0.5) * (w / n_cols)
    ys = (np.arange(n_rows) + 0.5) * (h / n_rows)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    X = X.flatten()[:N]
    Y = Y.flatten()[:N]
    r0 = min(w / n_cols, h / n_rows) / 2.0 * 0.9
    R = np.full(N, r0)
    return np.concatenate([X, Y, R, [w]])

def _random_seed(w, h):
    xs = np.random.uniform(0.1, w - 0.1, N)
    ys = np.random.uniform(0.1, h - 0.1, N)
    rs = np.full(N, min(w, h) * 0.05)
    return np.concatenate([xs, ys, rs, [w]])

def _force_seed(w, h, iterations=80):
    xs = np.random.uniform(0.1, w - 0.1, N)
    ys = np.random.uniform(0.1, h - 0.1, N)
    rs = np.full(N, min(w, h) * 0.05)
    for _ in range(iterations):
        for i in range(N):
            fx = fy = 0.0
            for j in range(N):
                if i == j:
                    continue
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < 1e-5:
                    fx += np.random.uniform(-1, 1)
                    fy += np.random.uniform(-1, 1)
                    continue
                f = 1.0 / d
                fx += f * dx
                fy += f * dy
            fx += 0.01 * (w / 2 - xs[i])
            fy += 0.01 * (h / 2 - ys[i])
            if xs[i] < rs[i]: fx -= 1.0
            if xs[i] > w - rs[i]: fx += 1.0
            if ys[i] < rs[i]: fy -= 1.0
            if ys[i] > h - rs[i]: fy += 1.0
            xs[i] = np.clip(xs[i] + 0.05 * fx, rs[i], w - rs[i])
            ys[i] = np.clip(ys[i] + 0.05 * fy, rs[i], h - rs[i])
    return np.concatenate([xs, ys, rs, [w]])

def _penalty_obj(x, lam=500.0):
    xs = x[:N]; ys = x[N:2 * N]; rs = x[2 * N:3 * N]; w = x[-1]; h = 2.0 - w
    if w <= 0.05 or h <= 0.05:
        return 1e6
    pen = 0.0
    # boundary violations (circle must fit)
    pen += np.sum(np.maximum(0.0, rs - xs)) ** 2
    pen += np.sum(np.maximum(0.0, rs - (w - xs))) ** 2
    pen += np.sum(np.maximum(0.0, rs - ys)) ** 2
    pen += np.sum(np.maximum(0.0, rs - (h - ys))) ** 2
    # overlap violations
    dx = xs[PI] - xs[PJ]
    dy = ys[PI] - ys[PJ]
    d = np.sqrt(dx * dx + dy * dy)
    rs_ = rs[PI] + rs[PJ]
    pen += np.sum(np.maximum(0.0, rs_ - d)) ** 2
    return -np.sum(rs) + lam * pen

def _is_valid(circles, tol=1e-6):
    n = circles.shape[0]
    xs = circles[:, 0]; ys = circles[:, 1]; rs = circles[:, 2]
    if np.any(rs < -tol):
        return False
    w = float(np.max(xs + rs))
    h = float(np.max(ys + rs))
    if 2 * (w + h) > 4.0 + tol:
        return False
    if np.any(xs < -rs - tol) or np.any(ys < -rs - tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx * dx + dy * dy < (rs[i] + rs[j]) ** 2 - tol:
                return False
    return True

def _final_circles(x):
    xs = x[:N].copy(); ys = x[N:2 * N].copy(); rs = x[2 * N:3 * N].copy()
    rs = np.maximum(rs, 0.0)
    w = float(np.max(xs + rs))
    h = float(np.max(ys + rs))
    # guarantee perimeter <= 4
    scale = 4.0 / (2.0 * (w + h))
    if scale < 1.0:
        xs *= scale; ys *= scale; rs *= scale
    # guarantee disjoint interiors (strict) by a tiny uniform shrink
    rs *= 0.999
    return np.column_stack((xs, ys, rs))

def _run_slsqp(x0, cons, bounds, maxiter, deadline, ftol=1e-9):
    try:
        res = minimize(_objective, x0, method='SLSQP', jac=_objective_grad,
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol, 'disp': 0})
        if time.time() > deadline:
            return None
        return res.x
    except Exception:
        return None

def _run_penalty(x0, deadline, lam=500.0, maxiter=400):
    try:
        res = minimize(lambda z: _penalty_obj(z, lam), x0, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'disp': 0})
        if time.time() > deadline:
            return None
        return res.x
    except Exception:
        return None

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    best_circles = None
    best_val = -np.inf
    best_w = 1.0

    def consider(x):
        nonlocal best_circles, best_val, best_w
        if x is None:
            return
        xs = x[:N]; ys = x[N:2 * N]; rs = x[2 * N:3 * N]; w = x[-1]
        if not (0.05 < w < 1.95):
            return
        h = 2.0 - w
        # clip to be safe
        xs = np.clip(xs, 0, w)
        ys = np.clip(ys, 0, h)
        rs = np.maximum(rs, 0.0)
        for i in range(N):
            max_r = min(xs[i], ys[i], w - xs[i], h - ys[i])
            rs[i] = min(rs[i], max_r)
        circ = np.column_stack((xs, ys, rs))
        if _is_valid(circ):
            val = float(np.sum(circ[:, 2]))
            if val > best_val:
                best_val = val
                best_circles = circ.copy()
                best_w = w

    cons = _cons()
    grid_configs = [(5, 5), (6, 4), (4, 6), (7, 4), (4, 7), (7, 3), (3, 7)]

    # ---- Phase 1: coarse search over aspect ratios (w as decision variable) ----
    aspect_ratios = [1.0, 1.2, 0.85, 1.3, 0.75, 1.1, 0.9, 1.4, 0.7, 1.5, 0.65]
    for ar in aspect_ratios:
        if time.time() > deadline - 25:
            break
        w = 2.0 * ar / (1 + ar)
        h = 2.0 - w
        if w <= 0.05 or h <= 0.05:
            continue
        bounds = [(0, w)] * N + [(0, h)] * N + [(1e-4, min(w, h) / 2)] * N + [(0.3, 1.7)]
        seeds = []
        for nc, nr in grid_configs:
            if nc * nr >= N:
                seeds.append(_grid_seed(w, h, nc, nr))
        seeds.append(_random_seed(w, h))
        seeds.append(_force_seed(w, h))
        for x0 in seeds:
            if time.time() > deadline - 20:
                break
            consider(_run_slsqp(x0, cons, bounds, maxiter=120, deadline=deadline))

    # ---- Phase 2: fine local search around the best aspect ratio ----
    if best_circles is not None:
        if time.time() < deadline - 15:
            base = np.concatenate([best_circles[:, 0], best_circles[:, 1],
                                   best_circles[:, 2], [best_w]])
            for dw in [-0.15, -0.08, -0.04, 0.0, 0.04, 0.08, 0.15]:
                if time.time() > deadline - 10:
                    break
                w = best_w + dw
                if w <= 0.1 or w >= 1.9:
                    continue
                h = 2.0 - w
                bounds = [(0, w)] * N + [(0, h)] * N + [(1e-4, min(w, h) / 2)] * N + [(max(0.1, w - 0.02), min(1.9, w + 0.02))]
                consider(_run_slsqp(base, cons, bounds, maxiter=150, deadline=deadline, ftol=1e-10))

    # ---- Phase 3: penalty-based search (lets w move freely, robust to bad starts) ----
    if time.time() < deadline - 5:
        for ar in [1.0, 1.2, 0.85, 1.3, 0.75]:
            if time.time() > deadline - 3:
                break
            w = 2.0 * ar / (1 + ar)
            h = 2.0 - w
            x0 = _random_seed(w, h)
            consider(_run_penalty(x0, deadline))
            if time.time() > deadline - 3:
                break

    # ---- Finalize ----
    if best_circles is None:
        # trivial valid fallback: 21 tiny circles in a row
        w = 1.0
        h = 1.0
        r = 0.01
        xs = np.linspace(r, w - r, N)
        ys = np.full(N, h / 2.0)
        best_circles = np.column_stack((xs, ys, np.full(N, r)))
    else:
        best_circles = _final_circles(np.concatenate(
            [best_circles[:, 0], best_circles[:, 1], best_circles[:, 2], [best_w]]))
        if not _is_valid(best_circles):
            w = 1.0; h = 1.0; r = 0.01
            xs = np.linspace(r, w - r, N)
            ys = np.full(N, h / 2.0)
            best_circles = np.column_stack((xs, ys, np.full(N, r)))

    return best_circles
