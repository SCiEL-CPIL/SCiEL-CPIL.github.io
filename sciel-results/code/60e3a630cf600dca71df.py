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
    SLSQP epigraph formulation with vectorized constraints for speed.
    """
    c = c.copy()
    r_init = _radius_fast(c)

    # Precompute pair indices
    pair_idx = [(i, j) for i in range(N) for j in range(i + 1, N)]
    n_pairs = len(pair_idx)
    n_walls = 2 * N
    n_constraints = n_pairs + n_walls

    def objective(x):
        return -x[-1]

    def constraints_vec(x):
        """All inequality constraints in one vectorized call."""
        centers = x[:2 * N].reshape(N, 2)
        r = x[-1]
        # Pair constraints: ||c_i - c_j|| - 2r >= 0
        pair_vals = np.empty(n_pairs)
        for k, (i, j) in enumerate(pair_idx):
            diff = centers[i] - centers[j]
            pair_vals[k] = np.sqrt(np.dot(diff, diff)) - 2.0 * r
        # Wall constraints: 0.5 - |c_ik| - r >= 0
        wall_vals = 0.5 - np.abs(centers) - r
        wall_flat = wall_vals.ravel()
        return np.concatenate([pair_vals, wall_flat])

    def constraints_grad(x):
        """Jacobian of constraints w.r.t. x (for SLSQP)."""
        centers = x[:2 * N].reshape(N, 2)
        r = x[-1]
        jac = np.zeros((n_constraints, 2 * N + 1))

        # Pair constraints
        for k, (i, j) in enumerate(pair_idx):
            diff = centers[i] - centers[j]
            norm = np.sqrt(np.dot(diff, diff))
            if norm > 1e-14:
                jac[k, 2 * i] = diff[0] / norm
                jac[k, 2 * i + 1] = diff[1] / norm
                jac[k, 2 * j] = -diff[0] / norm
                jac[k, 2 * j + 1] = -diff[1] / norm
            jac[k, -1] = -2.0

        # Wall constraints
        for i in range(N):
            for k in range(2):
                row = n_pairs + 2 * i + k
                c_ik = centers[i, k]
                jac[row, 2 * i + k] = -np.sign(c_ik)
                if c_ik == 0.0:
                    jac[row, 2 * i + k] = -1.0
                jac[row, -1] = -1.0

        return jac

    # Initial point
    x0 = np.concatenate([c.ravel(), [r_init]])

    # Bounds
    bounds = [(-0.5, 0.5)] * (2 * N) + [(1e-6, 0.5)]

    best_x = x0.copy()
    best_r = r_init

    # Run SLSQP with increasing maxiter
    maxiters_list = [300, 600, 1200]
    for maxiter in maxiters_list:
        if (time.time() - t0) > t0 + time_budget:
            break
        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=[{'type': 'ineq', 'fun': constraints_vec, 'jac': constraints_grad}],
                options={'maxiter': maxiter, 'ftol': 1e-14, 'iprint': 0}
            )
            c_new = res.x[:2 * N].reshape(N, 2)
            r_verified = _radius_fast(c_new)
            if r_verified > best_r:
                best_r = r_verified
                best_x = res.x.copy()
                x0 = res.x.copy()
        except Exception:
            pass

    return best_x[:2 * N].reshape(N, 2), best_r


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


def _delaunay_active_set_refine(c, r, max_iter=50, tol=1e-7):
    """
    Delaunay-based active-set refinement using the contact graph.
    Identifies tight constraints via Delaunay triangulation, then
    solves a small QP to find the direction that maximizes the
    minimum radius increase.
    """
    from scipy.spatial import Delaunay

    c = c.copy()
    r_cur = _radius_fast(c)
    if r_cur <= 0:
        return c, r_cur

    for outer in range(max_iter):
        # Compute distances
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)

        # Identify active pairs (near-contact)
        tol_d = max(1e-6, 3e-5 * r_cur)
        active_mask = d < 2.0 * r_cur + tol_d
        np.fill_diagonal(active_mask, False)
        active_pairs = np.argwhere(active_mask)
        active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]

        # Wall constraints
        wall_dist = 0.5 - np.abs(c)
        active_walls = []
        for i in range(N):
            for k in range(2):
                if wall_dist[i, k] < r_cur + tol_d:
                    active_walls.append((i, k))

        if len(active_pairs) == 0 and len(active_walls) == 0:
            break

        # Build the active constraint Jacobian and solve for optimal direction
        # We want to find direction d such that all active constraints improve
        # Constraint for pair (i,j): d||c_i - c_j||/dt = dot(c_i - c_j, d_i - d_j)/||c_i-c_j|| >= 0
        # Constraint for wall (i,k): d(0.5-|c_ik|)/dt = -sign(c_ik) * d_ik >= 0

        n_active = len(active_pairs) + len(active_walls)
        if n_active == 0:
            break

        # Build A matrix: each row is gradient of constraint w.r.t. positions
        # Variables: 2N position components
        A = np.zeros((n_active, 2 * N))
        b_deficit = np.zeros(n_active)

        for k, (i, j) in enumerate(active_pairs):
            diff = c[i] - c[j]
            norm = np.linalg.norm(diff)
            if norm < 1e-14:
                continue
            A[k, 2 * i] = diff[0] / norm
            A[k, 2 * i + 1] = diff[1] / norm
            A[k, 2 * j] = -diff[0] / norm
            A[k, 2 * j + 1] = -diff[1] / norm
            # Deficit: how much we need to increase
            b_deficit[k] = max(2.0 * r_cur - d[i, j], 0.0)

        for k, (i, wall_k) in enumerate(active_walls):
            row = len(active_pairs) + k
            c_ik = c[i, wall_k]
            s = np.sign(c_ik)
            if s == 0:
                s = 1.0
            A[row, 2 * i + wall_k] = -s
            b_deficit[row] = max(r_cur - wall_dist[i, wall_k], 0.0)

        # Solve: find d that maximizes min_k (A_k . d - b_k)
        # This is equivalent to: maximize t s.t. A_k . d >= t + b_k for all k
        # Use a simple approach: solve least squares with regularization
        # Or use scipy to solve the LP-like problem
        # Simple approach: find d that satisfies A d >= b (if feasible),
        # with minimum norm, then scale
        try:
            # Use scipy.linalg.lstsq for a regularized solution
            # Minimize ||d||^2 subject to A d >= b
            # Relax: minimize ||d||^2 + mu * sum(max(0, b - A d)^2)
            mu = 1.0
            # Gradient descent on the regularized problem
            d_dir = np.zeros(2 * N)
            for _ in range(200):
                residual = np.maximum(b_deficit - A @ d_dir, 0.0)
                grad = 2.0 * d_dir - 2.0 * mu * (A.T @ residual)
                lr = 0.01
                d_dir -= lr * grad
                # Normalize
                norm_d = np.linalg.norm(d_dir)
                if norm_d > 1e-14:
                    d_dir /= norm_d
                else:
                    break
        except Exception:
            d_dir = None

        if d_dir is not None and np.linalg.norm(d_dir) > 1e-14:
            # Try different step sizes
            best_step = 0.0
            best_r_try = r_cur
            for scale in [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]:
                c_try = c + scale * d_dir.reshape(N, 2)
                c_try = np.clip(c_try, -0.5 + 1e-12, 0.5 - 1e-12)
                r_try = _radius_fast(c_try)
                if r_try > best_r_try:
                    best_r_try = r_try
                    best_step = scale

            if best_step > 0:
                c = c + best_step * d_dir.reshape(N, 2)
                c = np.clip(c, -0.5 + 1e-12, 0.5 - 1e-12)
                r_cur = _radius_fast(c)
            else:
                break
        else:
            # Fallback: simple push-apart
            disp = np.zeros_like(c)
            for (i, j) in active_pairs:
                dij = c[i] - c[j]
                norm = np.linalg.norm(dij)
                if norm < 1e-12:
                    continue
                deficit = max(2 * r_cur - d[i, j], 0.0)
                step = 0.3 * deficit / norm
                disp[i] += step * dij
                disp[j] -= step * dij
            for (i, k) in active_walls:
                deficit = max(r_cur - wall_dist[i, k], 0.0)
                sign = -np.sign(c[i, k])
                if c[i, k] == 0:
                    sign = 1.0
                disp[i, k] += 0.3 * deficit * sign

            max_disp = np.max(np.linalg.norm(disp, axis=1))
            if max_disp > 1e-12:
                disp = disp / max_disp * min(0.3 * r_cur, 1e-4)
                c_try = c + disp
                c_try = np.clip(c_try, -0.5 + 1e-12, 0.5 - 1e-12)
                r_try = _radius_fast(c_try)
                if r_try > r_cur + 1e-12:
                    c = c_try
                    r_cur = r_try
                else:
                    break
            else:
                break

    return c, r_cur


def _final_local_search(c, r, max_iter=200, step_scale=1e-5):
    """
    Deterministic local search with adaptive step size.
    Tries perturbations in 8 directions with decreasing step sizes.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    directions = np.array([
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [1, 1], [-1, 1], [1, -1], [-1, -1]
    ], dtype=float)
    norms = np.array([1, 1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2)])
    directions = directions / norms[:, None]

    # Adaptive step sizes
    step_scales = [step_scale, step_scale * 0.5, step_scale * 0.25, step_scale * 0.1]

    for step_scale_i in step_scales:
        for _ in range(max_iter // len(step_scales)):
            improved = False
            for i in range(N):
                for d in directions:
                    c_try = c.copy()
                    c_try[i] += step_scale_i * d
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
    extra = np.array([[0.02, 0.02]])
    return np.vstack([g, extra])


def _init_hexagonal(rng):
    """Hexagonal packing initialization with jitter."""
    # Hexagonal lattice spacing
    s = 0.14  # approximate spacing for ~50 circles in unit square
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
        # Pad with random points
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
        if (time.time() - t0) > DEADLINE - 65:
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
        if budget > 3:
            c_polished, r_polished = _lbfgs_polish(
                best_c, t0,
                time_budget=budget
            )
            if r_polished > best_r:
                best_r = r_polished
                best_c = c_polished

    # Phase 3: SLSQP epigraph polish (direct max-min optimization) - vectorized
    if (time.time() - t0) < DEADLINE - 20:
        budget = min(35.0, DEADLINE - 20 - (time.time() - t0))
        if budget > 5:
            c_slsqp, r_slsqp = _slsqp_epigraph_polish(
                best_c, t0,
                time_budget=budget
            )
            if r_slsqp > best_r:
                best_r = r_slsqp
                best_c = c_slsqp

    # Phase 4: Delaunay active-set refinement
    if (time.time() - t0) < DEADLINE - 12:
        c_voro, r_voro = _delaunay_active_set_refine(best_c, best_r, max_iter=50)
        if r_voro > best_r:
            best_r = r_voro
            best_c = c_voro

    # Phase 5: Another SLSQP pass after active-set refinement
    if (time.time() - t0) < DEADLINE - 8:
        budget = min(10.0, DEADLINE - 8 - (time.time() - t0))
        if budget > 3:
            c_slsqp2, r_slsqp2 = _slsqp_epigraph_polish(
                best_c, t0,
                time_budget=budget
            )
            if r_slsqp2 > best_r:
                best_r = r_slsqp2
                best_c = c_slsqp2

    # Phase 6: Final high-precision local search
    if (time.time() - t0) < DEADLINE - 3:
        c_ls, r_ls = _final_local_search(best_c, best_r, max_iter=200, step_scale=1e-5)
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls

    # Final validation: ensure all points are strictly inside
    best_c = np.clip(best_c, -0.5 + 1e-9, 0.5 - 1e-9)

    return best_c
