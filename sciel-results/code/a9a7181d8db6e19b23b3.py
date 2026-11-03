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
    """Grow/relax a hard-disk packing: push overlapping circles apart and
    keep them inside the square. Returns a feasible layout for radius r."""
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
            # unit vectors from j to i
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
        # Adaptive timestep: reduce if large displacements
        max_disp = np.max(np.sqrt((disp ** 2).sum(axis=1)))
        if max_disp > 0.01:
            dt_eff = dt * min(1.0, 0.01 / max_disp)
        else:
            dt_eff = dt
        c = np.clip(c + dt_eff * disp, -0.5 + r, 0.5 - r)
    return c


def _contact_graph(c, r):
    """Identify active contacts via Delaunay: pairs with dist < 2r+eps and wall touches."""
    c = np.asarray(c, dtype=float)
    eps = 1e-9
    # Pairwise
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    pair_mask = d < 2 * r + eps
    # Walls
    wall_left = c[:, 0] < -0.5 + r + eps
    wall_right = c[:, 0] > 0.5 - r - eps
    wall_bottom = c[:, 1] < -0.5 + r + eps
    wall_top = c[:, 1] > 0.5 - r - eps
    return pair_mask, (wall_left, wall_right, wall_bottom, wall_top)


def _slsqp_epigraph(c0, r, time_budget=3.0):
    """SLSQP with epigraph variable t: maximize t s.t. d_ij >= 2t, walls >= t.
    Uses only active constraints (near-contact pairs) for efficiency."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    start = time.time()
    x0 = np.concatenate([c0.ravel(), [r]])

    # Identify active constraints: pairs with distance < 2r*1.05 and walls within r*1.05
    d = np.sqrt(((c0[:, None, :] - c0[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    active_pairs = [(i, j) for i in range(N) for j in range(i + 1, N) if d[i, j] < 2 * r * 1.05]
    
    # Wall constraints: all circles are potentially active
    active_walls = [(i, axis, sign) for i in range(N) for axis, sign in [(0, -1), (0, 1), (1, -1), (1, 1)]]

    def obj(x):
        return -x[-1]

    def obj_grad(x):
        g = np.zeros_like(x)
        g[-1] = -1.0
        return g

    cons = []
    # Active pairwise constraints
    for i, j in active_pairs:
        def c_pair(x, i=i, j=j):
            xi = x[2*i:2*i+2]
            xj = x[2*j:2*j+2]
            return np.sqrt(np.sum((xi - xj)**2)) - 2*x[-1]
        def c_pair_grad(x, i=i, j=j):
            g = np.zeros_like(x)
            xi = x[2*i:2*i+2]
            xj = x[2*j:2*j+2]
            dd = np.sqrt(np.sum((xi - xj)**2))
            if dd > 1e-12:
                diff = (xi - xj) / dd
                g[2*i:2*i+2] += diff
                g[2*j:2*j+2] -= diff
            g[-1] = -2.0
            return g
        cons.append({'type': 'ineq', 'fun': c_pair, 'jac': c_pair_grad})
    
    # Wall constraints
    for i, axis, sign in active_walls:
        def c_wall(x, i=i, axis=axis, sign=sign):
            val = x[2*i+axis]
            if sign > 0:
                return val + 0.5 - x[-1]
            else:
                return 0.5 - val - x[-1]
        def c_wall_grad(x, i=i, axis=axis, sign=sign):
            g = np.zeros_like(x)
            g[2*i+axis] = sign
            g[-1] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': c_wall, 'jac': c_wall_grad})

    try:
        res = minimize(obj, x0, jac=obj_grad, method='SLSQP',
                       bounds=[(-0.5, 0.5)] * (2 * N) + [(1e-6, 0.5)],
                       constraints=cons,
                       options={'maxiter': 300, 'ftol': 1e-15, 'iprint': 0})
        c = np.clip(res.x[:2*N].reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0
    return c


def _softmin_refine(c0, k, time_budget=3.0):
    """Smooth L-BFGS-B polish of a feasible layout to raise the min radius."""
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
        # Use log-sum-exp for numerical stability
        m = np.min(all_d)
        log_sw = m + np.log(np.sum(np.exp(k * (m - all_d))))
        val = log_sw / k

        weights = np.exp(-k * all_d)
        sw = weights.sum()
        grad_pts = np.zeros_like(pts)
        wp = weights[:N * N].reshape(N, N)
        mask = ~np.eye(N, dtype=bool)
        safe = np.where(mask, dists, 1.0)
        factors = np.where(mask, wp / sw / safe, 0.0)
        s = factors.sum(axis=1)
        wpts = (factors[..., None] * pts[None, :, :]).sum(axis=1)
        grad_pts += pts * s[:, None] - wpts
        ww = weights[N * N:].reshape(4, N) / sw
        grad_pts[:, 0] += -ww[0].sum() + ww[1].sum()
        grad_pts[:, 1] += -ww[2].sum() + ww[3].sum()
        return val, -grad_pts.ravel()

    try:
        res = minimize(obj_and_grad, c0.ravel(), jac=True, method='L-BFGS-B',
                       bounds=[(-0.5, 0.5)] * (2 * N),
                       options={'maxiter': 500, 'maxfun': 12000,
                                'ftol': 1e-18, 'gtol': 1e-14,
                                'iprint': 0})
        c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0
    return c


def _langevin_optimize(c0, r, time_budget=5.0, max_iter=50000, noise_scale=0.002):
    """Langevin dynamics: gradient descent on soft-min with thermal noise."""
    c = np.array(c0, dtype=float, copy=True)
    c = np.clip(c, -0.5, 0.5)
    start = time.time()
    k = 200.0  # temperature parameter for soft-min

    def soft_min_grad(pts):
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(dists, np.inf)
        dw = np.stack([
            0.5 - pts[:, 0], 0.5 + pts[:, 0],
            0.5 - pts[:, 1], 0.5 + pts[:, 1],
        ])
        all_d = np.concatenate([dists.ravel(), dw.ravel()])
        all_d = np.clip(all_d, 1e-8, 1.0)
        weights = np.exp(-k * all_d)
        sw = weights.sum()
        grad_pts = np.zeros_like(pts)
        wp = weights[:N * N].reshape(N, N)
        mask = ~np.eye(N, dtype=bool)
        safe = np.where(mask, dists, 1.0)
        factors = np.where(mask, wp / sw / safe, 0.0)
        s = factors.sum(axis=1)
        wpts = (factors[..., None] * pts[None, :, :]).sum(axis=1)
        grad_pts += pts * s[:, None] - wpts
        ww = weights[N * N:].reshape(4, N) / sw
        grad_pts[:, 0] += -ww[0].sum() + ww[1].sum()
        grad_pts[:, 1] += -ww[2].sum() + ww[3].sum()
        return grad_pts

    lr = 0.001
    for _ in range(max_iter):
        if time.time() - start > time_budget:
            break
        grad = soft_min_grad(c)
        noise = np.random.normal(0, noise_scale, c.shape)
        c = c - lr * grad + noise
        c = np.clip(c, -0.5, 0.5)
    return c


def construct():
    start_time = time.time()
    best = _grid_init()
    best_r = _true_score(best)

    inits = [_grid_init(), _hex_init()]
    rng = np.random.default_rng(12345)
    for _ in range(12):
        inits.append(rng.uniform(-0.42, 0.42, (N, 2)))

    # 1) Hard-disk radius expansion from several starts.
    for c0 in inits:
        if time.time() - start_time > DEADLINE - 20:
            break
        # estimate a starting radius from the layout
        r = _true_score(c0) * 0.95
        r = max(r, 0.05)
        lo, hi = 0.0, r
        cur = c0
        for _ in range(12):
            if time.time() - start_time > DEADLINE - 20:
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

    # 2) Langevin optimization phase for better exploration
    if time.time() - start_time < DEADLINE - 25:
        for c0 in [best] + inits[:3]:
            if time.time() - start_time > DEADLINE - 25:
                break
            c = _langevin_optimize(c0, best_r, time_budget=4.0, max_iter=30000, noise_scale=0.0015)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c

    # 3) Smooth soft-min polish from the best layout (and a couple others).
    candidates = [best]
    candidates.append(_force_relax(best, best_r, time_budget=1.5, max_iter=2000))
    for c0 in inits[:4]:
        candidates.append(_force_relax(c0, best_r, time_budget=1.5, max_iter=2000))

    # Progressive k values for better convergence
    for k in (50.0, 100.0, 200.0, 400.0):
        for c0 in candidates:
            if time.time() - start_time > DEADLINE - 10:
                break
            c = _softmin_refine(c0, k, time_budget=3.0)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c
                # Update candidates with new best
                candidates[0] = best
                candidates[1] = _force_relax(best, best_r, time_budget=1.0, max_iter=1000)

    # 4) SLSQP epigraph polish: exact constraint solve on the best layout
    if time.time() - start_time < DEADLINE - 8:
        c = _slsqp_epigraph(best, best_r, time_budget=5.0)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c
        # Second SLSQP pass with slightly larger target
        if time.time() - start_time < DEADLINE - 4:
            c = _slsqp_epigraph(best, best_r * 1.002, time_budget=3.0)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c

    # 5) Final soft-min polish with very high k
    if time.time() - start_time < DEADLINE - 2:
        c = _softmin_refine(best, 1500.0, time_budget=3.0)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    # 6) Final SLSQP polish after soft-min to capture any remaining gains
    if time.time() - start_time < DEADLINE - 1:
        c = _slsqp_epigraph(best, best_r, time_budget=1.5)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    return best
# EVOLVE-BLOCK-END
