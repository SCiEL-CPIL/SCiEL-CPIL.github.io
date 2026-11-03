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
DEADLINE = 140.0  # safety margin before 150s kill


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
    """Gradient of negative soft-min radius w.r.t. x (flattened centres). Vectorized."""
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
    weights = exps / Z

    grad = np.zeros((N, 2))

    # Wall gradients: term = 0.5 - |c_ik|, d/dc_ik = -sign(c_ik)
    sign_c = np.sign(c)
    grad += weights[:2 * N].reshape(N, 2) * (-sign_c)

    # Pair gradients: term_ij = ||c_i - c_j|| / 2
    # For each (i,j): contribution to grad[i] is w_ij * (c_i - c_j)/(2*|c_i-c_j|)
    #                contribution to grad[j] is w_ji * (c_j - c_i)/(2*|c_i-c_j|)
    # Combined: grad[i] += sum_j (w_ij + w_ji) * (c_i - c_j)/(2*|c_i-c_j|)
    diff = c[:, None, :] - c[None, :, :]  # (N, N, 2)
    dist_mat = d.copy()
    np.fill_diagonal(dist_mat, 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        g = diff / (2.0 * dist_mat[:, :, None])  # (N, N, 2)
    g[dist_mat < 1e-12] = 0.0

    w_pair = weights[2 * N:].reshape(N, N)  # (N, N)
    w_sym = w_pair + w_pair.T  # w_ij + w_ji
    np.fill_diagonal(w_sym, 0.0)

    # grad[i] += sum_j w_sym[i,j] * g[i,j]
    pair_grad = np.einsum('ij,ijk->k', w_sym, g)  # (N, 2)
    grad += pair_grad

    return grad.ravel()


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


def _voronoi_style_refine(c, r, max_iter=200, tol=1e-12):
    """
    Voronoi-style refinement: identify nearest-neighbor pairs and do
    targeted optimization on the kissing graph. For each pair of nearby
    circles, try to push them apart while respecting all other constraints.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    
    for iteration in range(max_iter):
        improved = False
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        wall = 0.5 - np.abs(c)
        
        # Find the most constrained pairs (closest to touching)
        # Sort all pairs by distance, try to improve the tightest ones
        pair_idx = np.triu_indices(N, k=1)
        pair_dists = d[pair_idx]
        # Get top 30 closest pairs
        n_try = min(30, len(pair_dists))
        top_idx = np.argsort(pair_dists)[:n_try]
        
        for idx in top_idx:
            i = pair_idx[0][idx]
            j = pair_idx[1][idx]
            dij = c[i] - c[j]
            dist = np.linalg.norm(dij)
            if dist < 1e-12:
                continue
            unit = dij / dist
            
            # Try pushing i and j apart slightly
            for push in [1e-6, 5e-6, 1e-5, 2e-5, 5e-5]:
                c_try = c.copy()
                c_try[i] += push * unit
                c_try[j] -= push * unit
                c_try = np.clip(c_try, -0.5 + 1e-12, 0.5 - 1e-12)
                r_try = _radius_fast(c_try)
                if r_try > r_cur + 1e-14:
                    c = c_try
                    r_cur = r_try
                    improved = True
                    break
        
        # Also try moving circles away from walls
        for i in range(N):
            for k in range(2):
                w = wall[i, k]
                if w < r_cur + 1e-3:
                    sign = np.sign(c[i, k])
                    if sign == 0:
                        sign = 1.0
                    for push in [1e-6, 5e-6, 1e-5]:
                        c_try = c.copy()
                        c_try[i, k] -= push * sign
                        c_try = np.clip(c_try, -0.5 + 1e-12, 0.5 - 1e-12)
                        r_try = _radius_fast(c_try)
                        if r_try > r_cur + 1e-14:
                            c = c_try
                            r_cur = r_try
                            improved = True
                            break
        
        if not improved:
            break
    
    return c, _radius_fast(c)


def _final_local_search(c, r, max_iter=500, step_scale=1e-5):
    """
    Deterministic local search: for each circle, try small perturbations
    in multiple directions, keeping the best improvement.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    
    # Generate directions: 24 angular directions
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    dirs = np.column_stack([np.cos(angles), np.sin(angles)])
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            for d in dirs:
                c_try = c.copy()
                c_try[i] += step_scale * d
                c_try = np.clip(c_try, -0.5 + 1e-12, 0.5 - 1e-12)
                r_try = _radius_fast(c_try)
                if r_try > r_cur + 1e-14:
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
    extra = np.array([[0.0, 0.0]])
    return np.vstack([g, extra])


def _init_jittered_8x8(rng):
    """8x8 grid, take 50, jitter."""
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    idx = rng.permutation(len(g))[:N]
    c = g[idx].copy()
    c += rng.uniform(-0.015, 0.015, c.shape)
    return c


def _init_hexagonal(rng):
    """Hexagonal close-packing initialization."""
    s = 0.12
    pts = []
    for row in range(8):
        y = -0.45 + row * s * np.sqrt(3) / 2
        offset = (s / 2) if (row % 2 == 1) else 0.0
        for col in range(8):
            x = -0.45 + col * s + offset
            if abs(x) <= 0.45 and abs(y) <= 0.45:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) > N:
        idx = rng.permutation(len(pts))[:N]
        pts = pts[idx]
    elif len(pts) < N:
        extra = rng.uniform(-0.4, 0.4, (N - len(pts), 2))
        pts = np.vstack([pts, extra])
    pts += rng.uniform(-0.01, 0.01, pts.shape)
    return np.clip(pts, -0.45, 0.45)


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
    while R <= R1 and (time.time() - t0) < DEADLINE - 20:
        c = _relax_hard_disk(c, R, iters_per_step, eta0, eta1)
        r = _radius_fast(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        R += step
    return best_c, best_r


def _slsqp_epigraph_polish(c, t0, time_budget=30.0):
    """
    SLSQP epigraph polish: maximize t subject to
      t <= ||c_i - c_j||/2  for all i<j
      t <= 0.5 - |c_ik|     for all i,k
    with c in [-0.5, 0.5].
    Uses scipy SLSQP with vectorized analytic gradients for fast convergence.
    """
    n = 2 * N
    nv = 2 * N + 1  # total variables: 2N centres + t
    n_pairs = N * (N - 1) // 2
    n_walls = 2 * N
    n_cons = n_pairs + n_walls

    # Precompute pair indices
    pair_i, pair_j = np.triu_indices(N, k=1)

    def _objective(xv):
        return -xv[2 * N]

    def _objective_grad(xv):
        g = np.zeros(nv)
        g[2 * N] = -1.0
        return g

    def _con_fun(xv):
        """All inequality constraint values (must be >= 0)."""
        c = xv[:2 * N].reshape(N, 2)
        t = xv[2 * N]
        # Wall constraints: 0.5 - |c_ik| - t
        wall_vals = 0.5 - np.abs(c) - t  # (N, 2)
        # Pair constraints: ||c_i - c_j||/2 - t
        ci = c[pair_i]
        cj = c[pair_j]
        dists = np.sqrt(np.sum((ci - cj) ** 2, axis=1))
        pair_vals = dists / 2.0 - t
        return np.concatenate([wall_vals.ravel(), pair_vals])

    def _con_jac(xv):
        """Full Jacobian matrix of shape (n_cons, nv)."""
        c = xv[:2 * N].reshape(N, 2)
        J = np.zeros((n_cons, nv))

        # Wall rows: 0.5 - |c_ik| - t >= 0
        # d/dc_ik = -sign(c_ik), d/dt = -1
        sign_c = np.sign(c)
        sign_c[sign_c == 0] = 1.0
        J[:n_walls, :2 * N] = -sign_c.ravel()[:, None] * np.eye(2 * N).ravel().reshape(2 * N, 2 * N)[:n_walls, :]
        # Simpler: for each wall constraint k (circle i, coord k), J[k, i*2+k] = -sign(c[i,k])
        J[:n_walls, :2 * N] = 0.0
        for idx in range(n_walls):
            ci = idx // 2
            ck = idx % 2
            s = -sign_c[ci, ck]
            J[idx, ci * 2 + ck] = s
        J[:n_walls, 2 * N] = -1.0

        # Pair rows: ||c_i - c_j||/2 - t >= 0
        ci = c[pair_i]
        cj = c[pair_j]
        diff = ci - cj  # (n_pairs, 2)
        dists = np.sqrt(np.sum(diff ** 2, axis=1))
        # Avoid division by zero
        safe_dists = np.maximum(dists, 1e-15)
        unit = diff / safe_dists[:, None]  # (n_pairs, 2)

        row_start = n_walls
        for idx in range(n_pairs):
            i = pair_i[idx]
            j = pair_j[idx]
            J[row_start + idx, i * 2] = unit[idx, 0] / 2.0
            J[row_start + idx, i * 2 + 1] = unit[idx, 1] / 2.0
            J[row_start + idx, j * 2] = -unit[idx, 0] / 2.0
            J[row_start + idx, j * 2 + 1] = -unit[idx, 1] / 2.0
            J[row_start + idx, 2 * N] = -1.0

        return J

    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]

    best_c = c.copy()
    best_r = _radius_fast(c)

    # Multiple restarts with different t initializations
    t_starts = [0.85, 0.92, 0.97, 1.0, 1.01, 1.02, 1.03]
    for ts in t_starts:
        if time.time() - t0 > t0 + time_budget:
            break
        x0 = np.zeros(nv)
        x0[:2 * N] = best_c.ravel()
        x0[2 * N] = _radius_fast(best_c) * ts
        if x0[2 * N] > 0.5:
            x0[2 * N] = 0.5
        try:
            res = minimize(
                _objective,
                x0,
                jac=_objective_grad,
                method='SLSQP',
                bounds=bounds,
                constraints=[{'type': 'ineq', 'fun': _con_fun, 'jac': _con_jac}],
                options={'maxiter': 300, 'ftol': 1e-14}
            )
            c_new = res.x[:2 * N].reshape(N, 2)
            r_new = _radius_fast(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
        except Exception:
            pass

    return best_c, best_r


def _lbfgs_polish(c, t0, time_budget=25.0):
    """
    L-BFGS-B polish using soft-min radius with adaptive decreasing temperature.
    """
    x = c.ravel()
    best_x = x.copy()
    best_r = _radius_fast(c)

    # Adaptive temperature schedule - start lower for better convergence
    temps = [50.0, 20.0, 10.0, 5.0, 2.5, 1.2, 0.6, 0.3, 0.15, 0.08]
    maxiter_per_temp = 500

    for t in temps:
        elapsed = time.time() - t0
        if elapsed > t0 + time_budget:
            break

        if t < 2.0:
            mi = maxiter_per_temp
        elif t < 10.0:
            mi = maxiter_per_temp * 3 // 4
        else:
            mi = maxiter_per_temp // 2

        try:
            res = minimize(
                lambda xx: _soft_min_radius_neg(xx, t),
                x,
                jac=lambda xx: _soft_min_radius_grad(xx, t),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': mi, 'gtol': 1e-14, 'ftol': 1e-18}
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

    for start in inits:
        if (time.time() - t0) > DEADLINE - 50:
            break
        # Phase 1: hard-disk radius expansion
        c, r = _grow_radius(start, 0.04, 0.072, 0.0005, 500, 0.02, 0.002, t0)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    if best_c is None:
        best_c = inits[0]
        best_r = _radius_fast(best_c)

    # Phase 2: L-BFGS-B soft-min polish
    if (time.time() - t0) < DEADLINE - 55:
        c_polished, r_polished = _lbfgs_polish(
            best_c, t0,
            time_budget=min(25.0, DEADLINE - 55 - (time.time() - t0))
        )
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished

    # Phase 2b: SLSQP epigraph polish (directly optimizes exact min radius)
    if (time.time() - t0) < DEADLINE - 35:
        c_slsqp, r_slsqp = _slsqp_epigraph_polish(
            best_c, t0,
            time_budget=min(30.0, DEADLINE - 35 - (time.time() - t0))
        )
        if r_slsqp > best_r:
            best_r = r_slsqp
            best_c = c_slsqp

    # Phase 3: Voronoi-style refinement
    if (time.time() - t0) < DEADLINE - 15:
        c_vs, r_vs = _voronoi_style_refine(best_c, best_r, max_iter=200)
        if r_vs > best_r:
            best_r = r_vs
            best_c = c_vs

    # Phase 3b: Second SLSQP pass after Voronoi refinement
    if (time.time() - t0) < DEADLINE - 10:
        c_slsqp2, r_slsqp2 = _slsqp_epigraph_polish(
            best_c, t0,
            time_budget=min(10.0, DEADLINE - 10 - (time.time() - t0))
        )
        if r_slsqp2 > best_r:
            best_r = r_slsqp2
            best_c = c_slsqp2

    # Phase 4: Final high-precision local search
    if (time.time() - t0) < DEADLINE - 5:
        c_ls, r_ls = _final_local_search(best_c, best_r, max_iter=300, step_scale=1e-5)
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls

        # One more round with even smaller steps
        if (time.time() - t0) < DEADLINE - 1:
            c_ls2, r_ls2 = _final_local_search(best_c, best_r, max_iter=200, step_scale=3e-6)
            if r_ls2 > best_r:
                best_r = r_ls2
                best_c = c_ls2

    # Final validation: ensure all points are strictly inside
    best_c = np.clip(best_c, -0.5 + 1e-9, 0.5 - 1e-9)

    # Verify validity
    r_check = _radius_fast(best_c)
    if r_check <= 0 or not np.isfinite(r_check):
        # Fallback to a known-good grid
        ticks = (np.arange(7) + 0.5) / 7 - 0.5
        g = np.array([(x, y) for y in ticks for x in ticks])
        extra = np.array([[0.0, 0.0]])
        best_c = np.vstack([g, extra])

    return best_c
