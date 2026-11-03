"""Combined packing program: contact-graph SLSQP epigraph + multi-stage refinement."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

DEADLINE = 145.0  # safety margin before the 150s kill


def _true_score(c):
    """Exact packing radius in double precision."""
    c = np.asarray(c, dtype=float)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    if np.any(c < -0.5) or np.any(c > 0.5):
        return 0.0
    r = float(np.min(0.5 - np.abs(c)))
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    r = min(r, float(d.min()) * 0.5)
    return r


def _force_relax(c, R, iters=3000, eta0=0.05, eta1=0.001):
    """Force-based hard-disk relaxation with wall forces."""
    c = np.array(c, dtype=float, copy=True)
    c = np.clip(c, -0.5 + R, 0.5 - R)
    for it in range(iters):
        t = it / max(iters - 1, 1)
        eta = eta0 + (eta1 - eta0) * t

        diff = c[:, None, :] - c[None, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, 1e-12)

        mag = np.maximum(2 * R - d, 0.0) / np.maximum(d, 1e-12)
        np.fill_diagonal(mag, 0.0)
        f = (mag[:, :, None] * diff).sum(axis=1)

        # Wall forces
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 1], 0.0)

        c += eta * f
        c = np.clip(c, -0.5 + R, 0.5 - R)
    return c


def _init_grid():
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = np.array([(x, y) for y in ticks for x in ticks])
    return pts[:N]


def _init_hex():
    m = 7
    pts = []
    for i in range(m):
        for j in range(m):
            x = (j + 0.5) / m - 0.5
            y = (i + 0.5) / m - 0.5
            if i % 2 == 1:
                x += 0.5 / m
            pts.append((x, y))
    pts = np.array(pts)
    return pts[:N]


def _init_random(seed=42):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, (N, 2))


def _slsqp_contact(c, r0, maxiter=500, time_budget=20.0, t_start=None):
    """
    SLSQP with contact-graph constraints.
    Only constrains pairs within a threshold distance.
    """
    if t_start is None:
        t_start = time.time()

    c = np.array(c, dtype=float, copy=True)
    r_init = max(r0, _true_score(c))
    if r_init <= 0:
        return c, r_init

    # Build contact graph: pairs within 2.5 * r
    threshold = 2.5 * r_init
    tree = cKDTree(c)
    pairs = tree.query_pairs(threshold)
    # Add wall contacts (all points)
    contact_pairs = list(pairs)

    def objective(x):
        return -x[-1]

    def objective_grad(x):
        g = np.zeros_like(x)
        g[-1] = -1.0
        return g

    constraints = []

    # Pairwise constraints for contact pairs only
    for i, j in contact_pairs:
        def pair_con(x, i=i, j=j):
            ci = x[2 * i:2 * i + 2]
            cj = x[2 * j:2 * j + 2]
            r = x[-1]
            d = np.sqrt(np.sum((ci - cj) ** 2))
            return d - 2 * r

        def pair_grad(x, i=i, j=j):
            g = np.zeros_like(x)
            ci = x[2 * i:2 * i + 2]
            cj = x[2 * j:2 * j + 2]
            diff = ci - cj
            d = np.sqrt(np.sum(diff ** 2))
            if d > 1e-12:
                g[2 * i:2 * i + 2] = diff / d
                g[2 * j:2 * j + 2] = -diff / d
            g[-1] = -2.0
            return g

        constraints.append({'type': 'ineq', 'fun': pair_con, 'jac': pair_grad})

    # Wall constraints
    for i in range(N):
        for k in range(2):
            def wall_con(x, i=i, k=k):
                c_ik = x[2 * i + k]
                r = x[-1]
                return 0.5 - np.abs(c_ik) - r

            def wall_grad(x, i=i, k=k):
                g = np.zeros_like(x)
                c_ik = x[2 * i + k]
                g[2 * i + k] = -np.sign(c_ik)
                g[-1] = -1.0
                return g

            constraints.append({'type': 'ineq', 'fun': wall_con, 'jac': wall_grad})

    x0 = np.concatenate([c.ravel(), [r_init]])
    bounds = [(-0.5, 0.5)] * (2 * N) + [(1e-6, 0.5)]

    best_x = x0.copy()
    best_r = r_init

    try:
        res = minimize(
            objective,
            x0,
            jac=objective_grad,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'disp': False}
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            r_new = res.x[-1]
            c_new = res.x[:2 * N].reshape(N, 2)
            r_verified = _true_score(c_new)
            if r_verified > best_r:
                best_r = r_verified
                best_x = res.x.copy()
    except Exception:
        pass

    return best_x[:2 * N].reshape(N, 2), best_r


def _soft_min_radius(x, t):
    """Soft-min of wall and pairwise distances."""
    c = x.reshape(N, 2)
    wall = 0.5 - np.abs(c)
    wall_flat = wall.ravel()

    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    pair_flat = (d / 2.0).ravel()

    all_terms = np.concatenate([wall_flat, pair_flat])
    m = all_terms.min()
    s = np.exp(-(all_terms - m) / t)
    return m - t * np.log(s.sum())


def _soft_min_grad(x, t):
    """Gradient of negative soft-min radius."""
    c = x.reshape(N, 2)
    wall = 0.5 - np.abs(c)
    wall_flat = wall.ravel()

    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    pair_flat = (d / 2.0).ravel()

    all_terms = np.concatenate([wall_flat, pair_flat])
    m = all_terms.min()
    exps = np.exp(-(all_terms - m) / t)
    Z = exps.sum()

    grad = np.zeros_like(c)
    sign_c = np.sign(c)
    w_exps = exps[:2 * N].reshape(N, 2)
    grad += (w_exps / Z) * (-sign_c)

    dist = d.copy()
    np.fill_diagonal(dist, 1.0)
    p_exps = exps[2 * N:].reshape(N, N)

    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        weights = p_exps[i, mask] / Z
        diffs = c[i] - c[mask]
        dists = dist[i, mask]
        contrib = np.sum(weights[:, None] * diffs / dists[:, None], axis=0)
        grad[i] += contrib * 0.5

    return -grad.ravel()


def _active_set_refine(c, r, maxiter=200, time_budget=10.0, t_start=None):
    """Refine using only active constraints (those at or near the minimum)."""
    if t_start is None:
        t_start = time.time()

    c = np.array(c, dtype=float, copy=True)
    r_init = max(r, _true_score(c))

    # Find active pairs: those with distance <= 2*r_init * 1.01
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)

    active_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            if d[i, j] <= 2 * r_init * 1.01:
                active_pairs.append((i, j))

    # Active walls
    active_walls = []
    for i in range(N):
        for k in range(2):
            w = 0.5 - np.abs(c[i, k])
            if w <= r_init * 1.01:
                active_walls.append((i, k))

    def objective(x):
        return -x[-1]

    def objective_grad(x):
        g = np.zeros_like(x)
        g[-1] = -1.0
        return g

    constraints = []

    for i, j in active_pairs:
        def pair_con(x, i=i, j=j):
            ci = x[2 * i:2 * i + 2]
            cj = x[2 * j:2 * j + 2]
            r = x[-1]
            return np.sqrt(np.sum((ci - cj) ** 2)) - 2 * r

        def pair_grad(x, i=i, j=j):
            g = np.zeros_like(x)
            ci = x[2 * i:2 * i + 2]
            cj = x[2 * j:2 * j + 2]
            diff = ci - cj
            d = np.sqrt(np.sum(diff ** 2))
            if d > 1e-12:
                g[2 * i:2 * i + 2] = diff / d
                g[2 * j:2 * j + 2] = -diff / d
            g[-1] = -2.0
            return g

        constraints.append({'type': 'ineq', 'fun': pair_con, 'jac': pair_grad})

    for i, k in active_walls:
        def wall_con(x, i=i, k=k):
            return 0.5 - np.abs(x[2 * i + k]) - x[-1]

        def wall_grad(x, i=i, k=k):
            g = np.zeros_like(x)
            g[2 * i + k] = -np.sign(x[2 * i + k])
            g[-1] = -1.0
            return g

        constraints.append({'type': 'ineq', 'fun': wall_con, 'jac': wall_grad})

    x0 = np.concatenate([c.ravel(), [r_init]])
    bounds = [(-0.5, 0.5)] * (2 * N) + [(1e-6, 0.5)]

    best_r = r_init
    best_x = x0.copy()

    try:
        res = minimize(
            objective, x0, jac=objective_grad,
            method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'disp': False}
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            r_new = res.x[-1]
            c_new = res.x[:2 * N].reshape(N, 2)
            r_verified = _true_score(c_new)
            if r_verified > best_r:
                best_r = r_verified
                best_x = res.x.copy()
    except Exception:
        pass

    return best_x[:2 * N].reshape(N, 2), best_r


def construct():
    start_time = time.time()

    # Stage 1: Initialization and force relaxation
    inits = [_init_grid(), _init_hex(), _init_random(42), _init_random(123), _init_random(999)]

    best_c = None
    best_r = 0.0

    for c0 in inits:
        if time.time() - start_time > DEADLINE - 50:
            break

        r = 0.04
        cur_c = c0.copy()
        for _ in range(8):
            cur_c = _force_relax(cur_c, r, iters=1000, eta0=0.03, eta1=0.001)
            s = _true_score(cur_c)
            if s >= r - 1e-9:
                r *= 1.08
            else:
                break

        s = _true_score(cur_c)
        if s > best_r:
            best_r = s
            best_c = cur_c.copy()

    if best_c is None:
        best_c = _init_grid()
        best_r = _true_score(best_c)

    # Stage 2: Soft-min polish with L-BFGS-B
    if time.time() - start_time < DEADLINE - 40:
        for t in [0.01, 0.005, 0.002]:
            if time.time() - start_time > DEADLINE - 40:
                break
            try:
                res = minimize(
                    lambda x: -_soft_min_radius(x, t),
                    best_c.ravel(),
                    jac=lambda x: _soft_min_grad(x, t),
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (2 * N),
                    options={'maxiter': 150, 'ftol': 1e-14, 'disp': False}
                )
                c_new = res.x.reshape(N, 2)
                s = _true_score(c_new)
                if s > best_r:
                    best_r = s
                    best_c = c_new
            except Exception:
                pass

    # Stage 3: Contact-graph SLSQP
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 15:
        budget = min(remaining - 5, 60)
        best_c, best_r = _slsqp_contact(best_c, best_r, maxiter=500, time_budget=budget)

    # Stage 4: Active-set refinement
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 10:
        budget = min(remaining - 3, 30)
        best_c, best_r = _active_set_refine(best_c, best_r, maxiter=300, time_budget=budget)

    # Stage 5: Final SLSQP with tighter tolerance
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 8:
        budget = min(remaining - 2, 20)
        best_c, best_r = _slsqp_contact(best_c, best_r, maxiter=300, time_budget=budget)

    # Final verification
    final_r = _true_score(best_c)
    if final_r > best_r:
        best_r = final_r

    return best_c
# EVOLVE-BLOCK-END
