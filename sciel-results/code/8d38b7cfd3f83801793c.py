"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

DEADLINE = 118.0  # safety margin before the 150s kill


def _true_score(c):
    """Exact packing radius in double precision."""
    c = np.asarray(c, dtype=float)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    if np.any(c < -0.5) or np.any(c > 0.5):
        return 0.0
    r = np.inf
    # wall distances
    w = np.min(np.stack([
        0.5 - c[:, 0], c[:, 0] + 0.5,
        0.5 - c[:, 1], c[:, 1] + 0.5,
    ]), axis=0)
    r = min(r, float(np.min(w)))
    # pairwise distances
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    r = min(r, float(np.min(d)) * 0.5)
    return r


def _grid_init():
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])  # 49 points
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _hex_init():
    m = 7
    pts = []
    for i in range(m):
        for j in range(m):
            pts.append(((j + 0.5) / m - 0.5, (i + 0.5) / m - 0.5))
    g = np.array(pts)
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _force_relax(c, r, time_budget=4.0, max_iter=20000):
    """Grow/relax a hard-disk packing with adaptive timestep."""
    c = np.array(c, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r, 0.5 - r)
    start = time.time()
    dt = 0.05
    for _ in range(max_iter):
        if time.time() - start > time_budget:
            break
        disp = np.zeros_like(c)
        # pairwise overlap repulsion
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
        # Adaptive timestep
        max_disp = np.max(np.sqrt((disp ** 2).sum(axis=1)))
        if max_disp > 0.01:
            dt_eff = dt * min(1.0, 0.01 / max_disp)
        else:
            dt_eff = dt
        c = np.clip(c + dt_eff * disp, -0.5 + r, 0.5 - r)
    return c


def _contact_graph(c, r, tol=1e-4):
    """Identify contact pairs and wall contacts using Delaunay triangulation."""
    try:
        tri = Delaunay(c)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    d = np.linalg.norm(c[a] - c[b])
                    if d < 2*r + tol:
                        edges.add((min(a,b), max(a,b)))
    except Exception:
        # Fallback: all pairs within 2r+tol
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        edges = set()
        for i in range(N):
            for j in range(i+1, N):
                if d[i,j] < 2*r + tol:
                    edges.add((i,j))
    return edges


def _slsqp_epigraph(c0, r0, time_budget=8.0, max_iter=200):
    """SLSQP epigraph: maximize t subject to distance >= 2t and wall >= t."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    x0 = np.concatenate([c0.ravel(), [r0]])
    n = 2*N + 1

    # Precompute pair indices
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]

    def obj(x):
        return -x[-1]  # minimize -t = maximize t

    def obj_grad(x):
        g = np.zeros(n)
        g[-1] = -1.0
        return g

    def constraints(x):
        pts = x[:2*N].reshape(N, 2)
        t = x[-1]
        cons = []
        # Pairwise: |c_i - c_j| >= 2t
        for i, j in pairs:
            di = pts[i] - pts[j]
            dist_sq = di[0]**2 + di[1]**2
            # Constraint: dist_sq - (2t)^2 >= 0
            val = dist_sq - 4*t*t
            # Gradient
            g = np.zeros(n)
            g[2*i] = 2*di[0]
            g[2*i+1] = 2*di[1]
            g[2*j] = -2*di[0]
            g[2*j+1] = -2*di[1]
            g[-1] = -8*t
            cons.append({'type': 'ineq', 'fun': lambda x, val=val, g=g: val, 'jac': lambda x, g=g: g})
        # Wall constraints: t <= 0.5 - |c_ik|
        for i in range(N):
            for axis in range(2):
                for sign in [1, -1]:
                    if sign > 0:
                        # t <= 0.5 - c[i, axis]
                        val = 0.5 - pts[i, axis] - t
                        g = np.zeros(n)
                        g[2*i+axis] = -1.0
                        g[-1] = -1.0
                    else:
                        # t <= 0.5 + c[i, axis]
                        val = 0.5 + pts[i, axis] - t
                        g = np.zeros(n)
                        g[2*i+axis] = 1.0
                        g[-1] = -1.0
                    cons.append({'type': 'ineq', 'fun': lambda x, val=val, g=g: val, 'jac': lambda x, g=g: g})
        return cons

    start = time.time()
    try:
        res = minimize(obj, x0, jac=obj_grad, method='SLSQP',
                       constraints=constraints,
                       bounds=[(-0.5, 0.5)] * (2*N) + [(0.0, 0.5)],
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'iprint': 0})
        c = np.clip(res.x[:2*N].reshape(N, 2), -0.5, 0.5)
        r = max(0.0, res.x[-1])
    except Exception:
        c = c0
        r = r0
    return c, r


def _softmin_refine(c0, k, time_budget=3.0):
    """Smooth L-BFGS-B polish with log-sum-exp stability."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    start = time.time()

    def obj_and_grad(x):
        pts = x.reshape(-1, 2)
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(dists, np.inf)
        dw = np.stack([
            0.5 - pts[:, 0], 0.5 + pts[:, 0],
            0.5 - pts[:, 1], 0.5 + pts[:, 1],
        ])
        all_d = np.concatenate([dists.ravel(), dw.ravel()])
        all_d = np.clip(all_d, 1e-8, 1.0)
        # Log-sum-exp for stability
        m = np.min(all_d)
        log_sw = m + np.log(np.sum(np.exp(k * (m - all_d))))
        val = log_sw / k

        weights = np.exp(-k * (all_d - m))
        sw = weights.sum()
        weights_norm = weights / sw

        grad_pts = np.zeros_like(pts)
        wp = weights_norm[:N * N].reshape(N, N)
        mask = ~np.eye(N, dtype=bool)
        safe = np.where(mask, dists, 1.0)
        factors = np.where(mask, wp / safe, 0.0)
        s = factors.sum(axis=1)
        wpts = (factors[..., None] * pts[None, :, :]).sum(axis=1)
        grad_pts += pts * s[:, None] - wpts
        ww = weights_norm[N * N:].reshape(4, N)
        grad_pts[:, 0] += -ww[0].sum() + ww[1].sum()
        grad_pts[:, 1] += -ww[2].sum() + ww[3].sum()
        return val, -grad_pts.ravel()

    try:
        res = minimize(obj_and_grad, c0.ravel(), jac=True, method='L-BFGS-B',
                       bounds=[(-0.5, 0.5)] * (2 * N),
                       options={'maxiter': 300, 'maxfun': 8000,
                                'ftol': 1e-16, 'gtol': 1e-12,
                                'iprint': 0})
        c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0
    return c


def construct():
    start_time = time.time()
    best = _grid_init()
    best_r = _true_score(best)

    inits = [_grid_init(), _hex_init()]
    rng = np.random.default_rng(12345)
    for _ in range(8):
        inits.append(rng.uniform(-0.42, 0.42, (N, 2)))

    # 1) Hard-disk radius expansion from several starts.
    for c0 in inits:
        if time.time() - start_time > DEADLINE - 25:
            break
        r = _true_score(c0) * 0.95
        r = max(r, 0.05)
        lo, hi = 0.0, r
        cur = c0
        for _ in range(10):
            if time.time() - start_time > DEADLINE - 25:
                break
            cur = _force_relax(cur, r, time_budget=2.0, max_iter=5000)
            s = _true_score(cur)
            if s >= r - 1e-9:
                lo = r
                r = r * 1.03
                hi = r
            else:
                hi = r
                if hi - lo < 1e-7:
                    break
                r = 0.5 * (lo + hi)
        s = _true_score(cur)
        if s > best_r:
            best_r, best = s, cur

    # 2) SLSQP epigraph refinement - primary method
    for c0 in [best] + inits[:3]:
        if time.time() - start_time > DEADLINE - 15:
            break
        cur = _force_relax(c0, best_r, time_budget=1.5, max_iter=2000)
        c, r = _slsqp_epigraph(cur, best_r, time_budget=6.0, max_iter=150)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    # 3) Soft-min polish as secondary
    candidates = [best]
    candidates.append(_force_relax(best, best_r, time_budget=1.0, max_iter=1500))

    for k in (100.0, 200.0, 400.0):
        for c0 in candidates:
            if time.time() - start_time > DEADLINE - 8:
                break
            c = _softmin_refine(c0, k, time_budget=2.5)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c
                candidates[0] = best
                candidates[1] = _force_relax(best, best_r, time_budget=0.8, max_iter=1000)

    # 4) Final SLSQP polish
    if time.time() - start_time < DEADLINE - 4:
        c, r = _slsqp_epigraph(best, best_r, time_budget=5.0, max_iter=100)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    # 5) Very fine soft-min
    if time.time() - start_time < DEADLINE - 1:
        c = _softmin_refine(best, 800.0, time_budget=3.0)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    return best
# EVOLVE-BLOCK-END
