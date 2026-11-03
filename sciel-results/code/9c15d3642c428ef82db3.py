"""Pack 50 equal circles in a unit square, maximize common radius.

Multi-stage approach:
1. Multiple initializations (7x7 grid, jittered 8x8, hexagonal, random)
2. Hard-disk radius expansion to reach near-optimal density
3. Multi-temperature L-BFGS-B soft-min polish with adaptive temperature schedule
4. Active-set refinement on the kissing graph
5. Final high-precision deterministic local search
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 115.0  # safety margin before 150s kill


def _radius_fast(c):
    """Compute max common radius for centres c (N,2) in [-0.5,0.5]^2."""
    wall_r = np.min(0.5 - np.abs(c))
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_r = d.min() / 2.0
    return min(wall_r, pair_r)


def _soft_min_radius(c, t):
    """
    Differentiable surrogate: soft-min of all pairwise half-distances
    and wall distances.  As t -> 0 this approaches the true min radius.
    """
    wall = 0.5 - np.abs(c)
    wall_flat = wall.ravel()

    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_terms = d / 2.0
    pair_flat = pair_terms.ravel()

    all_terms = np.concatenate([wall_flat, pair_flat])
    m = all_terms.min()
    s = np.exp(-(all_terms - m) / t)
    soft = m - t * np.log(s.sum())
    return soft


def _soft_min_radius_neg(x, t):
    c = x.reshape(N, 2)
    return -_soft_min_radius(c, t)


def _soft_min_radius_grad(x, t):
    """Gradient of negative soft-min radius w.r.t. x (flattened centres)."""
    c = x.reshape(N, 2)
    wall = 0.5 - np.abs(c)
    wall_flat = wall.ravel()

    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_terms = d / 2.0
    pair_flat = pair_terms.ravel()

    all_terms = np.concatenate([wall_flat, pair_flat])
    m = all_terms.min()
    exps = np.exp(-(all_terms - m) / t)
    Z = exps.sum()

    grad = np.zeros_like(c)

    # Wall gradients: term = 0.5 - |c_ik|, d/dc_ik = -sign(c_ik)
    sign_c = np.sign(c)
    w_grad = -sign_c.ravel()
    grad_flat = np.zeros(2 * N)
    grad_flat[:2 * N] += (exps[:2 * N] / Z) * w_grad

    # Pair gradients: term_ij = ||c_i - c_j|| / 2
    # d/dc_i = (c_i - c_j) / (2 * ||c_i - c_j||)
    # d/dc_j = (c_j - c_i) / (2 * ||c_i - c_j||)
    dist = d.copy()
    np.fill_diagonal(dist, 1.0)

    for i in range(N):
        for j in range(i + 1, N):
            idx_ij = i * N + j
            idx_ji = j * N + i
            w_ij = exps[2 * N + idx_ij] / Z
            w_ji = exps[2 * N + idx_ji] / Z
            total_w = w_ij + w_ji
            if total_w > 0:
                dij = c[i] - c[j]
                norm = dist[i, j]
                if norm > 1e-12:
                    g = dij / (2.0 * norm)
                    grad[i] += total_w * g
                    grad[j] -= total_w * g

    return grad_flat.reshape(N, 2)


def _relax_hard_disk(c, R, iters, eta0, eta1):
    """
    Hard-disk relaxation: push overlapping circles apart and keep them
    inside the square.  Vectorized.
    """
    c = c.copy()
    for it in range(iters):
        t = it / max(iters - 1, 1)
        eta = eta0 + (eta1 - eta0) * t
        diff = c[:, None, :] - c[None, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(d, 1e-12)
        mag = np.maximum(2 * R - d, 0.0) / np.maximum(d, 1e-12)
        np.fill_diagonal(mag, 0.0)
        f = np.sum(mag[:, :, None] * diff, axis=1)
        # Wall forces
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 1], 0.0)
        c += eta * f
        c = np.clip(c, -0.5, 0.5)
    return c


def _active_set_refine(c, r, max_iter=50, tol=1e-7):
    """
    Active-set refinement: identify kissing pairs and wall contacts,
    then solve a small QP to find a direction that increases the
    minimum radius.  This is a more principled version of the
    Voronoi-style refinement.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    if r_cur <= 0:
        return c, r_cur

    for _ in range(max_iter):
        # Identify active constraints
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)

        tol_d = max(1e-6, 1e-4 * r_cur)
        active_pairs = np.argwhere(d < 2 * r_cur + tol_d)
        active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]

        wall_dist = 0.5 - np.abs(c)
        active_walls = []
        for i in range(N):
            for k in range(2):
                if wall_dist[i, k] < r_cur + tol_d:
                    active_walls.append((i, k))

        if len(active_pairs) == 0 and len(active_walls) == 0:
            break

        # Build displacement vector
        disp = np.zeros_like(c)
        for (i, j) in active_pairs:
            dij = c[i] - c[j]
            norm = np.linalg.norm(dij)
            if norm < 1e-12:
                continue
            deficit = max(2 * r_cur - d[i, j], 0.0)
            step = 0.5 * deficit / norm
            disp[i] += step * dij
            disp[j] -= step * dij

        for (i, k) in active_walls:
            deficit = max(r_cur - wall_dist[i, k], 0.0)
            sign = -np.sign(c[i, k])
            if c[i, k] == 0:
                sign = 1.0
            disp[i, k] += 0.5 * deficit * sign

        # Normalize displacement
        max_disp = np.max(np.linalg.norm(disp, axis=1))
        if max_disp > 1e-12:
            disp = disp / max_disp * min(0.5 * r_cur, 1e-5)

        c_new = c + disp
        c_new = np.clip(c_new, -0.5, 0.5)
        r_new = _radius_fast(c_new)
        if r_new > r_cur + 1e-13:
            c = c_new
            r_cur = r_new
        else:
            # Try smaller step
            c_try = c + 0.1 * disp
            c_try = np.clip(c_try, -0.5, 0.5)
            r_try = _radius_fast(c_try)
            if r_try > r_cur + 1e-13:
                c = c_try
                r_cur = r_try
            else:
                break

    return c, r_cur


def _local_search_polish(c, r, max_iter=300, step_scale=1e-6):
    """
    High-precision local search: for each circle, try small perturbations
    in 8 directions, keeping the best improvement.  Repeat until no
    improvement is found.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    directions = np.array([
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [1, 1], [-1, 1], [1, -1], [-1, -1]
    ], dtype=float)
    directions /= np.sqrt(2)  # normalize diagonals

    for _ in range(max_iter):
        improved = False
        for i in range(N):
            for d in directions:
                c_try = c.copy()
                c_try[i] += step_scale * d
                c_try = np.clip(c_try, -0.5 + 1e-12, 0.5 - 1e-12)
                r_try = _radius_fast(c_try)
                if r_try > r_cur + 1e-13:
                    c = c_try
                    r_cur = r_try
                    improved = True
        if not improved:
            break
    return c, r_cur


def _init_7x7_plus():
    """7x7 grid (49 points) + 50th near centre."""
    ticks = (np.arange(7) + 0.5) / 7 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.02, 0.02]])
    return np.vstack([g, extra])


def _init_jittered_8x8(rng):
    """8x8 grid, take 50, jitter."""
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    idx = rng.permutation(len(g))[:N]
    c = g[idx].copy()
    c += rng.uniform(-0.02, 0.02, c.shape)
    return c


def _init_hexagonal(rng):
    """Hexagonal lattice initialization."""
    a = 1.0 / 7.0
    c = []
    for row in range(7):
        y = -0.5 + (row + 0.5) * a
        offset = 0.0 if row % 2 == 0 else a / 2.0
        for col in range(7):
            x = -0.5 + (col + 0.5) * a + offset
            c.append([x, y])
    c = np.array(c[:N])
    c += rng.uniform(-0.01, 0.01, c.shape)
    return c


def _init_random(rng):
    return rng.uniform(-0.45, 0.45, (N, 2))


def _grow_radius(start, R0, R1, step, iters_per_step, eta0, eta1, t0):
    """
    Gradually increase the target radius R from R0 to R1,
    running hard-disk relaxation at each step.
    """
    c = start.copy()
    best_c = c.copy()
    best_r = _radius_fast(c)
    R = R0
    while R <= R1 and (time.time() - t0) < DEADLINE - 10:
        c = _relax_hard_disk(c, R, iters_per_step, eta0, eta1)
        r = _radius_fast(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        R += step
    return best_c, best_r


def _slsqp_epigraph_polish(c, t0, time_budget=20.0):
    """
    SLSQP epigraph polish: maximize t subject to
      t <= 0.5 - |c_ik|  for all i,k
      t <= ||c_i - c_j|| / 2  for all i < j
      -0.5 <= c_ik <= 0.5
    This directly optimizes the exact min-radius objective.
    """
    n = 2 * N
    # Variables: [x_1, y_1, ..., x_N, y_N, t]
    x0 = np.concatenate([c.ravel(), [_radius_fast(c)]])
    best_x = x0.copy()
    best_r = _radius_fast(c)

    # Build constraint functions
    def _wall_constraints(xx):
        """t <= 0.5 - |c_ik| => 0.5 - |c_ik| - t >= 0"""
        c_ = xx[:n].reshape(N, 2)
        t_ = xx[n]
        return 0.5 - np.abs(c_) - t_

    def _pair_constraints(xx):
        """t <= ||c_i - c_j||/2 => ||c_i - c_j||/2 - t >= 0"""
        c_ = xx[:n].reshape(N, 2)
        t_ = xx[n]
        d = cdist(c_, c_)
        np.fill_diagonal(d, np.inf)
        return d / 2.0 - t_

    def _objective(xx):
        return -xx[n]  # maximize t => minimize -t

    def _objective_grad(xx):
        g = np.zeros_like(xx)
        g[n] = -1.0
        return g

    def _wall_grad(xx):
        c_ = xx[:n].reshape(N, 2)
        t_ = xx[n]
        grad = np.zeros((N * 2, N * 2))
        sign_c = np.sign(c_)
        # d(0.5 - |c_ik| - t)/dc_ik = -sign(c_ik)
        # d(0.5 - |c_ik| - t)/dt = -1
        for i in range(N):
            for k in range(2):
                grad[i * 2 + k, i * 2 + k] = -sign_c[i, k]
                grad[i * 2 + k, n] = -1.0
        return grad

    def _pair_grad(xx):
        c_ = xx[:n].reshape(N, 2)
        t_ = xx[n]
        d = cdist(c_, c_)
        np.fill_diagonal(d, 1.0)
        # Only compute for i < j pairs (upper triangle)
        grad = np.zeros((N * 2, N * 2))
        for i in range(N):
            for j in range(i + 1, N):
                dij = c_[i] - c_[j]
                norm = d[i, j]
                if norm < 1e-12:
                    continue
                # d(||c_i - c_j||/2 - t)/dc_i = dij / (2*norm)
                # d(||c_i - c_j||/2 - t)/dc_j = -dij / (2*norm)
                # d(||c_i - c_j||/2 - t)/dt = -1
                g = dij / (2.0 * norm)
                grad[i * 2, j * 2] = g[0]
                grad[i * 2, j * 2 + 1] = g[1]
                grad[i * 2 + 1, j * 2] = -g[0]
                grad[i * 2 + 1, j * 2 + 1] = -g[1]
                grad[j * 2, i * 2] = -g[0]
                grad[j * 2, i * 2 + 1] = -g[1]
                grad[j * 2 + 1, i * 2] = g[0]
                grad[j * 2 + 1, i * 2 + 1] = g[1]
                grad[i * 2, n] = -1.0
                grad[i * 2 + 1, n] = -1.0
        return grad

    # Run SLSQP with increasing time budget
    remaining = time_budget - (time.time() - t0)
    if remaining < 5.0:
        return c, best_r

    # Split into two phases: primary (60% budget) and final (40% budget)
    primary_budget = 0.6 * remaining
    final_budget = 0.4 * remaining

    for phase, budget in [(0, primary_budget), (1, final_budget)]:
        elapsed = time.time() - t0
        if elapsed > t0 + time_budget - 3.0:
            break

        # Tighter tolerance for final phase
        ftol = 1e-15 if phase == 1 else 1e-12
        maxiter = 300 if phase == 0 else 500

        constraints = [
            {'type': 'ineq', 'fun': _wall_constraints, 'jac': _wall_grad},
            {'type': 'ineq', 'fun': _pair_constraints, 'jac': _pair_grad},
        ]
        bounds = [(-0.5, 0.5)] * n + [(0.0, 0.5)]

        res = minimize(
            _objective,
            best_x,
            jac=_objective_grad,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': ftol, 'iprint': 0}
        )

        if res.x is not None and np.all(np.isfinite(res.x)):
            c_new = res.x[:n].reshape(N, 2)
            r_new = _radius_fast(c_new)
            if r_new > best_r:
                best_r = r_new
                best_x = res.x.copy()

    return best_x[:n].reshape(N, 2), best_r


def _voronoi_style_refine(c, r, max_iter=30, tol=1e-8):
    """
    Voronoi-style refinement: for each circle, compute its nearest neighbor
    and wall distance, then nudge it toward the center of its "free space"
    to increase the minimum radius.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    if r_cur <= 0:
        return c, r_cur

    for _ in range(max_iter):
        improved = False
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)

        for i in range(N):
            # Find nearest neighbor
            j = np.argmin(d[i])
            dij = c[i] - c[j]
            dist_ij = d[i, j]

            # Wall distances
            wall_d = 0.5 - np.abs(c[i])
            min_wall = np.min(wall_d)

            # Current min constraint for circle i
            min_constraint = min(dist_ij / 2.0, min_wall)

            if min_constraint > r_cur + tol:
                continue

            # Compute direction to increase min constraint
            # If pair is limiting, move away from nearest neighbor
            # If wall is limiting, move away from wall
            if dist_ij / 2.0 <= min_wall:
                # Pair limiting: move away from j
                if dist_ij > 1e-12:
                    dir_vec = dij / dist_ij
                else:
                    dir_vec = np.array([1.0, 0.0])
            else:
                # Wall limiting: move away from nearest wall
                dir_vec = np.zeros(2)
                for k in range(2):
                    if wall_d[k] == min_wall:
                        dir_vec[k] = -np.sign(c[i, k])
                        if c[i, k] == 0:
                            dir_vec[k] = 1.0
                norm = np.linalg.norm(dir_vec)
                if norm > 1e-12:
                    dir_vec /= norm
                else:
                    dir_vec = np.array([1.0, 0.0])

            # Try small steps
            for step in [1e-6, 5e-7, 1e-7, 5e-8]:
                c_try = c.copy()
                c_try[i] += step * dir_vec
                c_try = np.clip(c_try, -0.5 + 1e-12, 0.5 - 1e-12)
                r_try = _radius_fast(c_try)
                if r_try > r_cur + 1e-13:
                    c = c_try
                    r_cur = r_try
                    improved = True
                    break

        if not improved:
            break

    return c, r_cur


def _lbfgs_polish(c, t0, time_budget=15.0):
    """
    L-BFGS-B polish using soft-min radius with decreasing temperature.
    """
    x = c.ravel()
    best_x = x.copy()
    best_r = _radius_fast(c)

    temps = [50.0, 25.0, 10.0, 5.0, 2.0, 1.0, 0.5]
    for t in temps:
        if (time.time() - t0) > t0 + time_budget:
            break
        res = minimize(
            lambda xx: _soft_min_radius_neg(xx, t),
            x,
            jac=lambda xx: _soft_min_radius_grad(xx, t),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 200, 'gtol': 1e-12, 'ftol': 1e-15}
        )
        c_new = res.x.reshape(N, 2)
        r_new = _radius_fast(c_new)
        if r_new > best_r:
            best_r = r_new
            best_x = res.x.copy()
            x = res.x.copy()

    return best_x.reshape(N, 2), best_r


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)

    # Multiple initializations
    inits = [
        _init_7x7_plus(),
        _init_jittered_8x8(rng),
        _init_hexagonal(rng),
        _init_random(rng),
    ]

    best_c = None
    best_r = -1.0

    # Phase 1: hard-disk radius expansion for each init
    for start in inits:
        if (time.time() - t0) > DEADLINE - 30:
            break
        c, r = _grow_radius(start, 0.04, 0.072, 0.0005, 600, 0.015, 0.001, t0)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    if best_c is None:
        best_c = inits[0]
        best_r = _radius_fast(best_c)

    # Phase 2: SLSQP epigraph polish (primary)
    if (time.time() - t0) < DEADLINE - 30:
        c_sl, r_sl = _slsqp_epigraph_polish(
            best_c, t0,
            time_budget=min(25.0, DEADLINE - 30 - (time.time() - t0))
        )
        if r_sl > best_r:
            best_r = r_sl
            best_c = c_sl

    # Phase 3: Voronoi-style refinement
    if (time.time() - t0) < DEADLINE - 15:
        c_vor, r_vor = _voronoi_style_refine(best_c, best_r, max_iter=30)
        if r_vor > best_r:
            best_r = r_vor
            best_c = c_vor

    # Phase 4: SLSQP epigraph polish (final, tighter)
    if (time.time() - t0) < DEADLINE - 10:
        c_sl2, r_sl2 = _slsqp_epigraph_polish(
            best_c, t0,
            time_budget=min(15.0, DEADLINE - 10 - (time.time() - t0))
        )
        if r_sl2 > best_r:
            best_r = r_sl2
            best_c = c_sl2

    # Phase 5: Final high-precision local search
    if (time.time() - t0) < DEADLINE - 3:
        c_ls, r_ls = _local_search_polish(best_c, best_r, max_iter=300, step_scale=1e-6)
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls

    # Final validation: ensure all points are strictly inside
    best_c = np.clip(best_c, -0.5 + 1e-9, 0.5 - 1e-9)

    return best_c
