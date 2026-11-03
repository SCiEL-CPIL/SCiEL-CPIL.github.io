"""Pack 50 equal circles in a unit square, maximize common radius.

Multi-stage approach:
1. Multiple initializations (7x7 grid, jittered 8x8, hexagonal, random)
2. Hard-disk radius expansion to reach near-optimal density
3. Multi-temperature L-BFGS-B soft-min polish
4. SLSQP epigraph polish (direct max-min radius optimization)
5. Active-set refinement on the kissing graph
6. Final high-precision local search
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 145.0  # safety margin before 150s kill


def _radius_fast(c):
    """Compute max common radius for centres c (N,2) in [-0.5,0.5]^2."""
    wall_r = np.min(0.5 - np.abs(c))
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_r = d.min() / 2.0
    return min(wall_r, pair_r)


def _slsqp_epigraph_polish(c, t0, time_budget=30.0):
    """
    SLSQP epigraph formulation: maximize r subject to
    - pairwise distances >= 2r
    - wall distances >= r
    
    Variables: x = [c_1x, c_1y, ..., c_Nx, c_Ny, r]
    Objective: minimize -r
    Constraints:
      - For each pair (i,j): ||c_i - c_j|| - 2r >= 0
      - For each circle i and each wall k: (0.5 - |c_ik|) - r >= 0
      - Bounds: -0.5 <= c_ik <= 0.5, 0 < r < 0.5
    """
    c = c.copy()
    r_init = _radius_fast(c)
    
    def objective(x):
        return -x[-1]  # maximize r = x[-1]
    
    def constraint_pair(i, j):
        """Constraint: ||c_i - c_j|| >= 2r"""
        def _c(x):
            ci = x[2*i:2*i+2]
            cj = x[2*j:2*j+2]
            r = x[-1]
            return np.sqrt(np.sum((ci - cj)**2)) - 2*r
        return _c
    
    def constraint_wall(i, k):
        """Constraint: 0.5 - |c_ik| >= r"""
        def _c(x):
            c_ik = x[2*i + k]
            r = x[-1]
            return 0.5 - np.abs(c_ik) - r
        return _c
    
    # Build constraints
    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({'type': 'ineq', 'fun': constraint_pair(i, j)})
    for i in range(N):
        for k in range(2):
            constraints.append({'type': 'ineq', 'fun': constraint_wall(i, k)})
    
    # Initial point
    x0 = np.concatenate([c.ravel(), [r_init]])
    
    # Bounds: centres in [-0.5, 0.5], r in (0, 0.5)
    bounds = [(-0.5, 0.5)] * (2*N) + [(1e-6, 0.5)]
    
    best_x = x0.copy()
    best_r = r_init
    
    # Run SLSQP with multiple maxiter settings
    maxiters_list = [200, 500, 1000]
    for maxiter in maxiters_list:
        if (time.time() - t0) > t0 + time_budget:
            break
        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-12}
            )
            if res.success or res.fun < objective(x0):
                r_new = res.x[-1]
                c_new = res.x[:2*N].reshape(N, 2)
                # Verify the result
                r_verified = _radius_fast(c_new)
                if r_verified > best_r:
                    best_r = r_verified
                    best_x = res.x.copy()
                    x0 = res.x.copy()
        except Exception:
            pass
    
    return best_x[:2*N].reshape(N, 2), best_r


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


def _voronoi_style_refine(c, r, max_iter=30, tol=1e-6):
    """
    Voronoi-style active-set refinement: identify the 'kissing' graph
    (pairs at distance ~2r and circles touching walls), then for each
    active constraint compute a direction that increases the local
    minimum radius and take a small step.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    if r_cur <= 0:
        return c, r_cur

    for _ in range(max_iter):
        # Identify active constraints
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)

        tol_d = max(1e-5, 2e-4 * r_cur)
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

        # Build a displacement vector
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
            disp = disp / max_disp * min(0.5 * r_cur, 1e-4)

        c_new = c + disp
        c_new = np.clip(c_new, -0.5, 0.5)
        r_new = _radius_fast(c_new)
        if r_new > r_cur + 1e-12:
            c = c_new
            r_cur = r_new
        else:
            c_try = c + 0.1 * disp
            c_try = np.clip(c_try, -0.5, 0.5)
            r_try = _radius_fast(c_try)
            if r_try > r_cur + 1e-12:
                c = c_try
                r_cur = r_try
            else:
                break

    return c, r_cur


def _final_local_search(c, r, max_iter=200, step_scale=1e-5):
    """
    Deterministic local search: for each circle, try small perturbations
    in the 4 cardinal directions and the 4 diagonal directions, keeping
    the best improvement.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    directions = np.array([
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [1, 1], [-1, 1], [1, -1], [-1, -1]
    ], dtype=float)
    directions /= np.sqrt(2)

    for _ in range(max_iter):
        improved = False
        for i in range(N):
            for d in directions:
                c_try = c.copy()
                c_try[i] += step_scale * d
                c_try = np.clip(c_try, -0.5 + 1e-12, 0.5 - 1e-12)
                r_try = _radius_fast(c_try)
                if r_try > r_cur + 1e-12:
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


def _init_hexagonal(rng):
    """Hexagonal packing initialization with jitter."""
    s = 0.14
    c = []
    for row in range(8):
        y = -0.5 + s/2 + row * s * np.sqrt(3)/2
        if y > 0.5 - s/2:
            break
        offset = 0 if row % 2 == 0 else s/2
        for col in range(8):
            x = -0.5 + s/2 + col * s + offset
            if x <= 0.5 - s/2:
                c.append([x, y])
    c = np.array(c[:N])
    if len(c) < N:
        while len(c) < N:
            c = np.vstack([c, rng.uniform(-0.4, 0.4, 2)])
    c = c[:N]
    c += rng.uniform(-0.01, 0.01, c.shape)
    c = np.clip(c, -0.45, 0.45)
    return c


def _init_jittered_8x8(rng):
    """8x8 grid, take 50, jitter."""
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    idx = rng.permutation(len(g))[:N]
    c = g[idx].copy()
    c += rng.uniform(-0.02, 0.02, c.shape)
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


def _lbfgs_polish(c, t0, time_budget=10.0):
    """
    L-BFGS-B polish using soft-min radius with decreasing temperature.
    """
    x = c.ravel()
    best_x = x.copy()
    best_r = _radius_fast(c)

    temps = [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5]
    for t in temps:
        if (time.time() - t0) > t0 + time_budget:
            break
        try:
            res = minimize(
                lambda xx: _soft_min_radius_neg(xx, t),
                x,
                jac=lambda xx: _soft_min_radius_grad(xx, t),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14}
            )
            c_new = res.x.reshape(N, 2)
            r_new = _radius_fast(c_new)
            if r_new > best_r:
                best_r = r_new
                best_x = res.x.copy()
                x = res.x.copy()
        except Exception:
            pass

    return best_x.reshape(N, 2), best_r


def _interior_point_polish(c, t0, time_budget=25.0):
    """
    Interior-point style polish: use SLSQP with squared-distance constraints
    ||c_i - c_j||^2 >= 4r^2 for better conditioning of the active set.
    """
    c = c.copy()
    r_init = _radius_fast(c)
    
    def objective(x):
        return -x[-1]
    
    def constraint_pair_sq(i, j):
        """Constraint: ||c_i - c_j||^2 >= 4r^2"""
        def _c(x):
            ci = x[2*i:2*i+2]
            cj = x[2*j:2*j+2]
            r = x[-1]
            return np.sum((ci - cj)**2) - 4*r*r
        return _c
    
    def constraint_wall(i, k):
        """Constraint: 0.5 - |c_ik| >= r"""
        def _c(x):
            c_ik = x[2*i + k]
            r = x[-1]
            return 0.5 - np.abs(c_ik) - r
        return _c
    
    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({'type': 'ineq', 'fun': constraint_pair_sq(i, j)})
    for i in range(N):
        for k in range(2):
            constraints.append({'type': 'ineq', 'fun': constraint_wall(i, k)})
    
    x0 = np.concatenate([c.ravel(), [r_init]])
    bounds = [(-0.5, 0.5)] * (2*N) + [(1e-6, 0.5)]
    
    best_x = x0.copy()
    best_r = r_init
    
    for maxiter in [300, 600]:
        if (time.time() - t0) > t0 + time_budget:
            break
        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-13}
            )
            if res.success or res.fun < objective(x0):
                r_new = res.x[-1]
                c_new = res.x[:2*N].reshape(N, 2)
                r_verified = _radius_fast(c_new)
                if r_verified > best_r:
                    best_r = r_verified
                    best_x = res.x.copy()
                    x0 = res.x.copy()
        except Exception:
            pass
    
    return best_x[:2*N].reshape(N, 2), best_r


def _high_precision_polish(c, r, t0, time_budget=15.0):
    """
    Very fine-grained local search with adaptive step sizes and
    multiple random restarts of the search direction.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    
    # Start with fine steps, then try even finer
    for step_scale in [5e-6, 2e-6, 1e-6, 5e-7]:
        if (time.time() - t0) > t0 + time_budget:
            break
        directions = np.array([
            [1, 0], [-1, 0], [0, 1], [0, -1],
            [1, 1], [-1, 1], [1, -1], [-1, -1],
            [1, 2], [-1, 2], [1, -2], [-1, -2],
            [2, 1], [-2, 1], [2, -1], [-2, -1]
        ], dtype=float)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        
        for _ in range(50):
            if (time.time() - t0) > t0 + time_budget:
                break
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

    # Phase 1: Hard-disk radius expansion on all initializations
    for start in inits:
        if (time.time() - t0) > DEADLINE - 60:
            break
        c, r = _grow_radius(start, 0.04, 0.072, 0.0005, 800, 0.015, 0.001, t0)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    if best_c is None:
        best_c = inits[0]
        best_r = _radius_fast(best_c)

    # Phase 2: L-BFGS-B soft-min polish
    if (time.time() - t0) < DEADLINE - 50:
        budget = min(15.0, DEADLINE - 50 - (time.time() - t0))
        if budget > 5:
            c_polished, r_polished = _lbfgs_polish(
                best_c, t0,
                time_budget=budget
            )
            if r_polished > best_r:
                best_r = r_polished
                best_c = c_polished

    # Phase 3: SLSQP epigraph polish
    if (time.time() - t0) < DEADLINE - 30:
        budget = min(20.0, DEADLINE - 30 - (time.time() - t0))
        if budget > 5:
            c_slsqp, r_slsqp = _slsqp_epigraph_polish(
                best_c, t0,
                time_budget=budget
            )
            if r_slsqp > best_r:
                best_r = r_slsqp
                best_c = c_slsqp

    # Phase 4: Interior-point style polish with squared constraints
    if (time.time() - t0) < DEADLINE - 20:
        budget = min(15.0, DEADLINE - 20 - (time.time() - t0))
        if budget > 5:
            c_ip, r_ip = _interior_point_polish(
                best_c, t0,
                time_budget=budget
            )
            if r_ip > best_r:
                best_r = r_ip
                best_c = c_ip

    # Phase 5: Voronoi-style active-set refinement
    if (time.time() - t0) < DEADLINE - 12:
        c_voro, r_voro = _voronoi_style_refine(best_c, best_r, max_iter=50)
        if r_voro > best_r:
            best_r = r_voro
            best_c = c_voro

    # Phase 6: High-precision final polish
    if (time.time() - t0) < DEADLINE - 5:
        budget = min(10.0, DEADLINE - 5 - (time.time() - t0))
        if budget > 2:
            c_hp, r_hp = _high_precision_polish(best_c, best_r, t0, time_budget=budget)
            if r_hp > best_r:
                best_r = r_hp
                best_c = c_hp

    # Final validation
    best_c = np.clip(best_c, -0.5 + 1e-9, 0.5 - 1e-9)

    return best_c
