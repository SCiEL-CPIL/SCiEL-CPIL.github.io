"""Pack 50 equal circles in a unit square, maximize common radius.

Uses a multi-stage approach:
1. Multiple initializations (grid, jittered grid, random)
2. Hard-disk radius expansion to get close to optimal
3. Active-set QCQP-style refinement: identify the kissing graph
   (pairs at distance ~2r and wall contacts), then solve a local
   optimization where only active constraints are enforced as equalities,
   maximizing r subject to those equalities and the remaining inequalities.
4. Multi-temperature L-BFGS-B soft-min polish
5. High-precision final local search
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


def _active_set_qcqp_refine(c, r, max_iter=50, tol=1e-7):
    """
    Active-set QCQP refinement: identify the kissing graph (pairs at
    distance ~2r and wall contacts), then for each active constraint
    compute a direction that increases the local minimum radius while
    maintaining the active constraints.  This exploits the fact that at
    optimality, the packing is determined by a rigid graph of touching
    circles.
    
    We use a Newton-style step: solve for displacements that maintain
    active constraints as equalities while maximizing r.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    if r_cur <= 0:
        return c, r_cur

    for outer in range(max_iter):
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

        # Build the constraint Jacobian for active constraints.
        # For each active pair (i,j): constraint is ||c_i - c_j||^2 = (2r)^2
        #   gradient w.r.t. c_i: 2*(c_i - c_j), w.r.t. c_j: 2*(c_j - c_i)
        # For each active wall (i,k): constraint is c_ik = ±(0.5 - r)
        #   gradient: unit vector in coordinate k for circle i

        n_constraints = len(active_pairs) + len(active_walls)
        if n_constraints == 0:
            break

        # We want to find a displacement delta such that:
        # 1. Active constraints remain satisfied (Jacobian * delta = 0 for equality part)
        # 2. The radius increases: we want to move in a direction that
        #    increases the minimum of all constraints.
        
        # Alternative approach: use a weighted combination of constraint
        # gradients to find an ascent direction for r.
        
        # For each circle, compute the net force from active constraints.
        disp = np.zeros_like(c)
        
        # Pair forces: for active pair (i,j) at distance d_ij < 2r,
        # the "pressure" pushes them apart.
        for (i, j) in active_pairs:
            dij = c[i] - c[j]
            norm = np.linalg.norm(dij)
            if norm < 1e-12:
                continue
            # Weight by how close to contact
            deficit = max(2 * r_cur - d[i, j], 0.0)
            weight = 1.0 + deficit / max(r_cur, 1e-10)
            step = weight * 0.5 * deficit / norm
            disp[i] += step * dij
            disp[j] -= step * dij

        # Wall forces: for active wall (i,k), push inward.
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

        # Try the step
        c_new = c + disp
        c_new = np.clip(c_new, -0.5, 0.5)
        r_new = _radius_fast(c_new)
        
        if r_new > r_cur + 1e-12:
            c = c_new
            r_cur = r_new
        else:
            # Try smaller steps
            found = False
            for scale in [0.5, 0.25, 0.1, 0.05]:
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

        # Build a displacement vector that pushes each circle away from
        # its active constraints.
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

        # Normalize displacement to avoid overshooting
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
            # Try a smaller step
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
    in the 4 cardinal directions and the 4 diagonal directions, keeping
    the best improvement.  Repeat until no improvement is found.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    directions = np.array([
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [1, 1], [-1, 1], [1, -1], [-1, -1]
    ], dtype=float)
    directions /= np.sqrt(2)  # normalize diagonals

    # Adaptive step size
    step = step_scale
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            for d in directions:
                c_try = c.copy()
                c_try[i] += step * d
                c_try = np.clip(c_try, -0.5 + 1e-12, 0.5 - 1e-12)
                r_try = _radius_fast(c_try)
                if r_try > r_cur + 1e-13:
                    c = c_try
                    r_cur = r_try
                    improved = True
        if not improved:
            # Try smaller step
            step *= 0.5
            if step < 1e-9:
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


def _init_random(rng):
    return rng.uniform(-0.45, 0.45, (N, 2))


def _init_hexagonal(rng):
    """Hexagonal packing initialization."""
    c = np.zeros((N, 2))
    dx = 0.12
    dy = dx * np.sqrt(3) / 2
    row = 0
    col = 0
    idx = 0
    while idx < N:
        x = (col + 0.5 * (row % 2)) * dx - 0.5
        y = row * dy - 0.5
        if -0.45 <= x <= 0.45 and -0.45 <= y <= 0.45:
            c[idx] = [x, y]
            idx += 1
        col += 1
        if col > 8:
            col = 0
            row += 1
    # Add small jitter to break symmetry
    c += rng.uniform(-0.005, 0.005, c.shape)
    return c


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

    temps = [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2]
    for t in temps:
        if (time.time() - t0) > t0 + time_budget:
            break
        res = minimize(
            lambda xx: _soft_min_radius_neg(xx, t),
            x,
            jac=lambda xx: _soft_min_radius_grad(xx, t),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 400, 'gtol': 1e-12, 'ftol': 1e-15}
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

    for start in inits:
        if (time.time() - t0) > DEADLINE - 30:
            break
        # Phase 1: hard-disk radius expansion
        c, r = _grow_radius(start, 0.04, 0.072, 0.0005, 800, 0.015, 0.001, t0)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    if best_c is None:
        best_c = inits[0]
        best_r = _radius_fast(best_c)

    # Phase 2: L-BFGS-B soft-min polish
    if (time.time() - t0) < DEADLINE - 25:
        c_polished, r_polished = _lbfgs_polish(
            best_c, t0,
            time_budget=min(20.0, DEADLINE - 25 - (time.time() - t0))
        )
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished

    # Phase 3: Active-set QCQP refinement
    if (time.time() - t0) < DEADLINE - 15:
        c_qcqp, r_qcqp = _active_set_qcqp_refine(best_c, best_r, max_iter=50)
        if r_qcqp > best_r:
            best_r = r_qcqp
            best_c = c_qcqp

    # Phase 4: Voronoi-style active-set refinement
    if (time.time() - t0) < DEADLINE - 10:
        c_voro, r_voro = _voronoi_style_refine(best_c, best_r, max_iter=30)
        if r_voro > best_r:
            best_r = r_voro
            best_c = c_voro

    # Phase 5: Another L-BFGS pass with lower temperatures
    if (time.time() - t0) < DEADLINE - 8:
        c_polished2, r_polished2 = _lbfgs_polish(
            best_c, t0,
            time_budget=min(8.0, DEADLINE - 8 - (time.time() - t0))
        )
        if r_polished2 > best_r:
            best_r = r_polished2
            best_c = c_polished2

    # Phase 6: Final high-precision local search
    if (time.time() - t0) < DEADLINE - 3:
        c_ls, r_ls = _final_local_search(best_c, best_r, max_iter=300, step_scale=1e-5)
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls

    # Final validation: ensure all points are strictly inside
    best_c = np.clip(best_c, -0.5 + 1e-9, 0.5 - 1e-9)

    return best_c
