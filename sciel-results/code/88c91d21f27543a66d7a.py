"""Pack 50 equal circles in a unit square, maximize common radius.

Multi-stage approach:
1. Multiple initializations (7x7 grid, jittered 8x8, hexagonal, random)
2. Hard-disk radius expansion to reach near-optimal density
3. Multi-temperature L-BFGS-B soft-min polish with adaptive temperature schedule
4. Active-set QCQP refinement on the kissing graph with weighted forces
5. Second L-BFGS-B polish pass after active-set refinement
6. Final high-precision deterministic local search with adaptive steps
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 116.0  # safety margin before 150s kill


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

    grad_flat = np.zeros(2 * N)

    # Wall gradients: term = 0.5 - |c_ik|, d/dc_ik = -sign(c_ik)
    sign_c = np.sign(c)
    w_grad = -sign_c.ravel()
    grad_flat[:2 * N] += (exps[:2 * N] / Z) * w_grad

    # Pair gradients: term_ij = ||c_i - c_j|| / 2
    dist = d.copy()
    np.fill_diagonal(dist, 1.0)

    for i in range(N):
        for j in range(i + 1, N):
            idx_ij = i * N + j
            idx_ji = j * N + i
            w_ij = exps[2 * N + idx_ij] / Z
            w_ji = exps[2 * N + idx_ji] / Z
            total_w = w_ij + w_ji
            if total_w > 1e-15:
                dij = c[i] - c[j]
                norm = dist[i, j]
                if norm > 1e-12:
                    g = dij / (2.0 * norm)
                    grad_flat[i * 2] += total_w * g[0]
                    grad_flat[i * 2 + 1] += total_w * g[1]
                    grad_flat[j * 2] -= total_w * g[0]
                    grad_flat[j * 2 + 1] -= total_w * g[1]

    return grad_flat


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


def _active_set_qcqp_refine(c, r, max_iter=80, tol=1e-7):
    """
    Active-set QCQP refinement: identify the kissing graph (pairs at
    distance ~2r and wall contacts), then for each active constraint
    compute a weighted displacement that increases the local minimum
    radius while maintaining the active constraints.
    
    Uses deficit-based weighting: constraints that are more violated
    (or closer to being violated) get higher weight, mimicking the
    pressure in a rigid contact network.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    if r_cur <= 0:
        return c, r_cur

    for outer in range(max_iter):
        if (time.time()) > DEADLINE - 10:
            break

        # Identify active constraints
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)

        tol_d = max(1e-6, 2e-4 * r_cur)
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

        # Build displacement with weighted forces from active constraints
        disp = np.zeros_like(c)
        
        # Pair forces: for active pair (i,j), push apart with deficit weighting
        for (i, j) in active_pairs:
            dij = c[i] - c[j]
            norm = np.linalg.norm(dij)
            if norm < 1e-12:
                continue
            # Deficit: how much the distance is below 2r
            deficit = max(2 * r_cur - d[i, j], 0.0)
            # Weight: higher for more violated constraints
            weight = 1.0 + deficit / max(r_cur, 1e-10)
            step = weight * 0.5 * deficit / max(norm, 1e-12)
            disp[i] += step * dij
            disp[j] -= step * dij

        # Wall forces: for active wall (i,k), push inward with deficit weighting
        for (i, k) in active_walls:
            deficit = max(r_cur - wall_dist[i, k], 0.0)
            sign = -np.sign(c[i, k])
            if c[i, k] == 0:
                sign = 1.0
            disp[i, k] += deficit * sign

        # Normalize displacement to avoid overshooting
        max_disp = np.max(np.linalg.norm(disp, axis=1))
        if max_disp < 1e-12:
            break
        disp = disp / max_disp * min(0.3 * r_cur, 5e-5)

        # Try the step with backtracking
        c_new = c + disp
        c_new = np.clip(c_new, -0.5, 0.5)
        r_new = _radius_fast(c_new)
        
        if r_new > r_cur + 1e-12:
            c = c_new
            r_cur = r_new
        else:
            # Try smaller steps
            found = False
            for scale in [0.5, 0.25, 0.1, 0.05, 0.02]:
                c_try = c + scale * disp
                c_try = np.clip(c_try, -0.5, 0.5)
                r_try = _radius_fast(c_try)
                if r_try > r_cur + 1e-12:
                    c = c_try
                    r_cur = r_try
                    found = True
                    break
            if not found:
                break

    return c, r_cur


def _voronoi_style_refine(c, r, max_iter=40, tol=1e-6):
    """
    Voronoi-style active-set refinement: identify kissing graph,
    compute directions that increase the local minimum radius.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    if r_cur <= 0:
        return c, r_cur

    for _ in range(max_iter):
        if (time.time()) > DEADLINE - 8:
            break

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

        disp = np.zeros_like(c)
        for (i, j) in active_pairs:
            dij = c[i] - c[j]
            norm = np.linalg.norm(dij)
            if norm < 1e-12:
                continue
            deficit = max(2 * r_cur - d[i, j], 0.0)
            step = 0.5 * deficit / max(norm, 1e-12)
            disp[i] += step * dij
            disp[j] -= step * dij

        for (i, k) in active_walls:
            deficit = max(r_cur - wall_dist[i, k], 0.0)
            sign = -np.sign(c[i, k])
            if c[i, k] == 0:
                sign = 1.0
            disp[i, k] += 0.5 * deficit * sign

        max_disp = np.max(np.linalg.norm(disp, axis=1))
        if max_disp < 1e-12:
            break
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


def _final_local_search(c, r, max_iter=300, step_scale=1e-5):
    """
    Deterministic local search: for each circle, try small perturbations
    in multiple directions, keeping the best improvement.
    Uses adaptive step scaling based on current radius.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    
    # 24 angular directions for finer search
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    all_dirs = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Also add cardinal and diagonal directions
    extra_dirs = np.array([
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [1, 1], [-1, 1], [1, -1], [-1, -1]
    ], dtype=float)
    extra_dirs /= np.sqrt(2)
    all_dirs = np.vstack([all_dirs, extra_dirs])

    for it in range(max_iter):
        if (time.time()) > DEADLINE - 3:
            break
        
        improved = False
        for i in range(N):
            for d in all_dirs:
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


def _slsqp_epigraph_polish(c, r_init, time_budget=20.0):
    """
    SLSQP polish using epigraph formulation.
    Maximizes r subject to non-overlap and wall constraints.
    """
    def fun(x):
        return -x[-1]

    def jac(x):
        g = np.zeros_like(x)
        g[-1] = -1.0
        return g

    def con(x):
        c = x[:2*N].reshape(N, 2)
        r = x[-1]
        i_idx, j_idx = np.triu_indices(N, 1)
        d_sq = np.sum((c[i_idx] - c[j_idx])**2, axis=1)
        pair_vals = d_sq - 4*r*r
        wall_vals = 0.5 - np.abs(c).ravel() - r
        return np.concatenate([pair_vals, wall_vals])

    def jac_con(x):
        c = x[:2*N].reshape(N, 2)
        r = x[-1]
        i_idx, j_idx = np.triu_indices(N, 1)
        num_pairs = len(i_idx)
        num_walls = 2*N
        J = np.zeros((num_pairs + num_walls, 2*N + 1))
        
        diff = c[i_idx] - c[j_idx]
        for k in range(num_pairs):
            i = i_idx[k]
            j = j_idx[k]
            J[k, 2*i:2*i+2] = 2.0 * diff[k]
            J[k, 2*j:2*j+2] = -2.0 * diff[k]
            J[k, -1] = -8.0 * r
            
        for i in range(N):
            for k in range(2):
                idx = num_pairs + 2*i + k
                J[idx, -1] = -1.0
                val = c[i, k]
                if val > 0:
                    J[idx, 2*i+k] = -1.0
                elif val < 0:
                    J[idx, 2*i+k] = 1.0
                else:
                    J[idx, 2*i+k] = 0.0
        return J

    x0 = np.concatenate([c.ravel(), [0.99 * r_init]])
    bounds = [(-0.5, 0.5)] * (2*N) + [(0.01, 0.5)]
    
    best_x = x0
    best_r = _radius_fast(c)
    
    start_time = time.time()
    
    try:
        res = minimize(
            fun, x0, jac=jac, 
            method='SLSQP', 
            bounds=bounds,
            constraints=[{'type': 'ineq', 'fun': con, 'jac': jac_con}],
            options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
        )
        r_new = res.x[-1]
        c_new = res.x[:2*N].reshape(N, 2)
        r_actual = _radius_fast(c_new)
        if r_actual > best_r:
            best_r = r_actual
            best_x = res.x.copy()
    except Exception:
        pass
        
    # Second run if time permits
    if time.time() - start_time < time_budget * 0.5:
        try:
            res2 = minimize(
                fun, best_x, jac=jac, 
                method='SLSQP', 
                bounds=bounds,
                constraints=[{'type': 'ineq', 'fun': con, 'jac': jac_con}],
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            r_new2 = res2.x[-1]
            c_new2 = res2.x[:2*N].reshape(N, 2)
            r_actual2 = _radius_fast(c_new2)
            if r_actual2 > best_r:
                best_r = r_actual2
                best_x = res2.x.copy()
        except Exception:
            pass

    return best_x[:2*N].reshape(N, 2), best_r


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

    # Phase 1: hard-disk radius expansion for each initialization
    for start in inits:
        if (time.time() - t0) > DEADLINE - 35:
            break
        c, r = _grow_radius(start, 0.04, 0.072, 0.0005, 600, 0.02, 0.002, t0)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    if best_c is None:
        best_c = inits[0]
        best_r = _radius_fast(best_c)

    # Phase 2: SLSQP epigraph polish
    if (time.time() - t0) < DEADLINE - 25:
        c_slsqp, r_slsqp = _slsqp_epigraph_polish(
            best_c, best_r,
            time_budget=min(25.0, DEADLINE - 25 - (time.time() - t0))
        )
        if r_slsqp > best_r:
            best_r = r_slsqp
            best_c = c_slsqp

    # Phase 3: Active-set QCQP refinement
    if (time.time() - t0) < DEADLINE - 15:
        c_as, r_as = _active_set_qcqp_refine(best_c, best_r, max_iter=80)
        if r_as > best_r:
            best_r = r_as
            best_c = c_as

    # Phase 4: Second SLSQP epigraph polish after active-set refinement
    if (time.time() - t0) < DEADLINE - 10:
        c_slsqp2, r_slsqp2 = _slsqp_epigraph_polish(
            best_c, best_r,
            time_budget=min(10.0, DEADLINE - 10 - (time.time() - t0))
        )
        if r_slsqp2 > best_r:
            best_r = r_slsqp2
            best_c = c_slsqp2

    # Phase 5: Voronoi-style refinement
    if (time.time() - t0) < DEADLINE - 8:
        c_voro, r_voro = _voronoi_style_refine(best_c, best_r, max_iter=40)
        if r_voro > best_r:
            best_r = r_voro
            best_c = c_voro

    # Phase 6: Final high-precision local search
    if (time.time() - t0) < DEADLINE - 4:
        c_ls, r_ls = _final_local_search(best_c, best_r, max_iter=300, step_scale=1e-5)
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls

        # One more round with even smaller steps
        if (time.time() - t0) < DEADLINE - 2:
            c_ls2, r_ls2 = _final_local_search(best_c, best_r, max_iter=200, step_scale=3e-6)
            if r_ls2 > best_r:
                best_r = r_ls2
                best_c = c_ls2

    # Final validation: ensure all points are strictly inside
    best_c = np.clip(best_c, -0.5 + 1e-9, 0.5 - 1e-9)

    return best_c
