"""Circle packing in a unit square: maximize common radius for 50 equal circles.
Combines multi-start hard-disk relaxation with SLSQP epigraph polish and
Langevin dynamics refinement."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 50
DEADLINE = 118.0  # safety margin before 150s kill


def _score(pos):
    """Compute the maximum valid radius for given positions."""
    pos = np.asarray(pos, dtype=np.float64)
    if pos.shape != (N, 2):
        return -1.0
    if not np.all(np.isfinite(pos)):
        return -1.0
    # Wall constraint
    wall = 0.5 - np.abs(pos).max(axis=1)
    r_wall = wall.min()
    if r_wall < 0:
        return -1.0
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    r_pair = (dist.min() / 2.0)
    return min(r_wall, r_pair)


def _relax(c, R, iters, eta0, eta1, t0=None, deadline=None):
    """
    Relaxation step: push overlapping circles apart and keep them inside the square.
    """
    c = c.copy()
    n = len(c)
    for it in range(iters):
        if t0 is not None and deadline is not None:
            if time.time() - t0 > deadline - 3:
                break
        t = it / max(iters, 1)
        eta = eta0 + (eta1 - eta0) * t

        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, 1e-12)
        mag = np.maximum(2 * R - d, 0.0) / d
        np.fill_diagonal(mag, 0.0)
        f = (mag[:, :, np.newaxis] * diff).sum(1)
        # Walls
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 1], 0.0)
        c += eta * f
        c = np.clip(c, -0.5, 0.5)
    return c


def _grow(start, R0, R1, step, iters, eta0, eta1, t0, deadline):
    """
    Grow the radius from R0 to R1 in steps, relaxing at each step.
    """
    c = start.copy()
    best = (c.copy(), _score(c))
    R = R0
    while R <= R1 and (time.time() - t0) < deadline - 5:
        c = _relax(c, R, iters, eta0, eta1, t0, deadline)
        r = _score(c)
        if r > best[1]:
            best = (c.copy(), r)
        R += step
        if (time.time() - t0) > deadline - 5:
            break
    return best


def _init_grid_7x7():
    """7x7 grid with one extra circle at center."""
    ticks = (np.arange(7) + 0.5) / 7 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    return np.vstack([g, extra])


def _init_jitter8(rng):
    """8x8 grid, pick 50, add jitter."""
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    idx = rng.permutation(len(g))[:N]
    c = g[idx].copy()
    c += rng.uniform(-0.03, 0.03, c.shape)
    return c


def _init_rand(rng):
    """Random uniform positions."""
    return rng.uniform(-0.45, 0.45, (N, 2))


def _init_hex():
    """Hexagonal lattice starting position."""
    positions = []
    spacing = 0.14
    row_spacing = spacing * np.sqrt(3) / 2
    for row in range(8):
        for col in range(7):
            x = (col - 3) * spacing + (0.5 * spacing if row % 2 else 0)
            y = (row - 3.5) * row_spacing
            positions.append([x, y])
    positions = np.array(positions[:N])
    max_abs = np.abs(positions).max()
    if max_abs > 0.45:
        positions = positions * (0.45 / max_abs)
    return positions


def _init_grid_8x8():
    """8x8 grid, pick 50."""
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    return g[:N].copy()


def _langevin_refine(c, r, t0, deadline, steps=5000, T0=0.001, T1=0.0001, lr=0.002):
    """
    Langevin dynamics refinement: add stochastic kicks to escape local minima.
    """
    c = c.copy()
    best_c = c.copy()
    best_r = r
    rng = np.random.default_rng(999)

    for step in range(steps):
        if time.time() - t0 > deadline - 5:
            break
        t = step / steps
        T = T0 + (T1 - T0) * t
        lr_t = lr * (1.0 - 0.5 * t)

        # Compute forces (same as relaxation)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, 1e-12)
        mag = np.maximum(2 * r - d, 0.0) / d
        np.fill_diagonal(mag, 0.0)
        f = (mag[:, :, np.newaxis] * diff).sum(1)
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - r), 0.0) + np.maximum((-0.5 + r) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - r), 0.0) + np.maximum((-0.5 + r) - c[:, 1], 0.0)

        # Langevin update: deterministic + stochastic
        noise = T * rng.standard_normal(c.shape)
        c += lr_t * f + noise
        c = np.clip(c, -0.5, 0.5)

        r_new = _score(c)
        if r_new > best_r:
            best_r = r_new
            best_c = c.copy()

    return best_c, best_r


def _slsqp_epigraph(c0, r_init, time_budget=15.0, max_iter=300, ftol=1e-14):
    """
    SLSQP with linear epigraph formulation:
    maximize t subject to:
      t <= |c_i - c_j| / 2  for all i < j
      t <= 0.5 - c[i,k]     for all i, k (upper walls)
      t <= c[i,k] + 0.5     for all i, k (lower walls)
    """
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    r_init = max(r_init, 0.01)

    def neg_t(x):
        return -x[2 * N]

    def neg_t_grad(x):
        g = np.zeros(2 * N + 1)
        g[2 * N] = -1.0
        return g

    cons = []
    # Pairwise constraints: |c_i - c_j|/2 - t >= 0
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
                t = x[2 * N]
                diff = ci - cj
                d = np.sqrt(np.sum(diff ** 2))
                if d < 1e-12:
                    g[2 * i:2 * i + 2] = 0
                    g[2 * j:2 * j + 2] = 0
                    g[2 * N] = -1.0
                    return g
                g[2 * i:2 * i + 2] = diff / (2.0 * d)
                g[2 * j:2 * j + 2] = -diff / (2.0 * d)
                g[2 * N] = -1.0
                return g

            cons.append({'type': 'ineq', 'fun': pair_con, 'jac': pair_con_grad})

    # Wall constraints
    for i in range(N):
        for axis in range(2):
            def wall_con(x, i=i, axis=axis):
                t = x[2 * N]
                return 0.5 - x[2 * i + axis] - t

            def wall_con_grad(x, i=i, axis=axis):
                g = np.zeros(2 * N + 1)
                g[2 * i + axis] = -1.0
                g[2 * N] = -1.0
                return g

            cons.append({'type': 'ineq', 'fun': wall_con, 'jac': wall_con_grad})

            def wall_con2(x, i=i, axis=axis):
                t = x[2 * N]
                return x[2 * i + axis] + 0.5 - t

            def wall_con2_grad(x, i=i, axis=axis):
                g = np.zeros(2 * N + 1)
                g[2 * i + axis] = 1.0
                g[2 * N] = -1.0
                return g

            cons.append({'type': 'ineq', 'fun': wall_con2, 'jac': wall_con2_grad})

    x0 = np.concatenate([c0.ravel(), [r_init]])
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]

    try:
        res = minimize(neg_t, x0, jac=neg_t_grad, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': max_iter, 'ftol': ftol, 'iprint': 0})
        if res.success or (res.x is not None and np.all(np.isfinite(res.x))):
            c = np.clip(res.x[:2 * N].reshape(N, 2), -0.5, 0.5)
            r = float(res.x[2 * N])
            if r > r_init * 0.99:
                return c, r
    except Exception:
        pass

    return c0, r_init


def _softmin_refine(c0, k, time_budget=3.0):
    """Smooth L-BFGS-B polish of a feasible layout to raise the min radius."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)

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
        weights = np.exp(-k * all_d)
        sw = weights.sum()
        val = -np.log(sw) / k
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
                       options={'maxiter': 150, 'maxfun': 4000,
                                'ftol': 1e-14, 'gtol': 1e-10,
                                'iprint': 0})
        c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0
    return c


def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)

    # Multiple diverse initializations
    inits = [
        _init_grid_7x7(),
        _init_jitter8(rng),
        _init_hex(),
        _init_grid_8x8(),
        _init_rand(np.random.default_rng(42)),
        _init_rand(np.random.default_rng(77)),
    ]

    best = None
    best_r = -1.0

    # Phase 1: Grow radius from each starting configuration
    n_inits = len(inits)
    time_per_init = (DEADLINE - (time.time() - t0) - 40) / n_inits  # reserve 40s for refinement

    for i, start in enumerate(inits):
        remaining = DEADLINE - (time.time() - t0)
        if remaining < 15:
            break

        budget = min(time_per_init, remaining * 0.85 / (n_inits - i))
        budget = max(budget, 3.0)

        # Grow from 0.05 to 0.0715
        c, r = _grow(start, 0.05, 0.0715, 0.0004, 1500, 0.02, 0.002, t0, DEADLINE)

        if r > best_r:
            best_r = r
            best = c.copy()

        # Check time
        if time.time() - t0 > DEADLINE - 45:
            break

    if best is None:
        best = _init_grid_7x7()
        best_r = _score(best)

    # Phase 2: Soft-min refinement on the best configuration
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 8:
        c_soft = _softmin_refine(best, k=50.0, time_budget=5.0)
        r_soft = _score(c_soft)
        if r_soft > best_r:
            best_r = r_soft
            best = c_soft

    # Phase 3: SLSQP epigraph polish
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 12:
        c_slq, r_slq = _slsqp_epigraph(best, best_r, time_budget=12.0, max_iter=200, ftol=1e-14)
        if r_slq > best_r:
            best_r = r_slq
            best = c_slq

    # Phase 4: Langevin refinement to escape local minima
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 10:
        c, r = _langevin_refine(best, best_r, t0, DEADLINE, steps=6000, T0=0.0012, T1=0.00005, lr=0.0025)
        if r > best_r:
            best_r = r
            best = c

    # Phase 5: Another SLSQP pass after Langevin
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 10:
        c_slq, r_slq = _slsqp_epigraph(best, best_r, time_budget=10.0, max_iter=150, ftol=1e-14)
        if r_slq > best_r:
            best_r = r_slq
            best = c_slq

    # Phase 6: Final fine-tune relaxation at current radius
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 5:
        r_cur = _score(best)
        c = _relax(best, r_cur, 2000, 0.01, 0.0005, t0, DEADLINE)
        r_new = _score(c)
        if r_new >= r_cur:
            best = c
            best_r = r_new

    # Final check
    if not np.all(np.isfinite(best)) or best.shape != (N, 2):
        best = _init_grid_7x7()

    return best
