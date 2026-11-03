"""Pack 50 equal circles in a unit square, maximize common radius.

Uses a multi-stage approach:
1. Multiple initializations (grid, jittered grid, hexagonal, random)
2. Hard-disk radius expansion to get close to optimal
3. Active-set refinement on the kissing graph
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

    grad = np.zeros((N, 2))

    # Wall gradients: d(0.5 - |c_ik|)/d(c_ik) = -sign(c_ik)
    sign_c = np.sign(c)
    grad += (exps[:2 * N] / Z).reshape(N, 2) * (-sign_c)

    # Pair gradients: for each pair (i,j), contribution to grad[i] += w * (c[i]-c[j])/(2*d_ij)
    # where w = (exp_ij + exp_ji)/Z
    w_pair = (exps[2 * N:]).reshape(N, N) / Z
    # Symmetrize: w_pair[i,j] should equal w_pair[j,i] since d[i,j]=d[j,i]
    w_pair = (w_pair + w_pair.T) * 0.5

    # diff[i,j] = c[i] - c[j], shape (N, N, 2)
    diff = c[:, None, :] - c[None, :, :]
    # d[i,j] = ||diff[i,j]||
    d_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(d_mat, 1.0)

    # grad[i] = sum_j w_pair[i,j] * diff[i,j] / (2 * d_mat[i,j])
    # = sum_j w_pair[i,j] / (2 * d_mat[i,j]) * (c[i] - c[j])
    coeff = w_pair / (2.0 * d_mat)
    np.fill_diagonal(coeff, 0.0)
    # grad[i] = coeff[i,:] @ diff[i,:]  (sum over j)
    grad += np.sum(coeff[:, :, None] * diff, axis=1)

    return grad


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
    Active-set refinement: identify the kissing graph (pairs at
    distance ~2r and wall contacts), then for each active constraint
    compute a direction that increases the local minimum radius while
    maintaining the active constraints.
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

        # Build a displacement vector that pushes each circle away from
        # its active constraints.
        disp = np.zeros_like(c)
        for (i, j) in active_pairs:
            dij = c[i] - c[j]
            norm = np.linalg.norm(dij)
            if norm < 1e-12:
                continue
            deficit = max(2 * r_cur - d[i, j], 0.0)
            weight = 1.0 + deficit / max(r_cur, 1e-10)
            step = weight * 0.5 * deficit / norm
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


def _slsqp_epigraph_polish(c, t0, time_budget=15.0):
    """
    SLSQP epigraph formulation: maximize r subject to
      ||c_i - c_j||^2 >= 4 r^2  for all i < j
      0.5 - |c_i_k| >= r        for all i, k
    Decision variables: x = [c_1x, c_1y, ..., c_Nx, c_Ny, r]
    Objective: minimize -r (i.e., maximize r)
    Constraints are smooth quadratic/linear, so SLSQP handles them well.
    """
    from scipy.optimize import minimize
    import warnings
    warnings.filterwarnings('ignore')

    x0 = np.concatenate([c.ravel(), [_radius_fast(c)]])
    best_x = x0.copy()
    best_r = x0[-1]

    # Precompute pair indices for speed
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((i, j))
    pairs = np.array(pairs)
    n_pairs = len(pairs)
    n_wall = 2 * N

    def objective(x):
        return -x[-1]

    def objective_grad(x):
        g = np.zeros(2 * N + 1)
        g[-1] = -1.0
        return g

    # Constraint: for each pair (i,j): (x_i - x_j)^2 >= 4 r^2
    # In SLSQP form: (x_i - x_j)^2 - 4*r^2 >= 0  ->  type 'ineq' means >= 0
    def pair_constraints(x):
        ci = x[:2*N].reshape(N, 2)
        r = x[-1]
        ci_i = ci[pairs[:, 0]]
        ci_j = ci[pairs[:, 1]]
        diffs = ci_i - ci_j
        dist2 = np.sum(diffs ** 2, axis=1)
        return dist2 - 4.0 * r * r

    def pair_constraints_jac(x):
        ci = x[:2*N].reshape(N, 2)
        r = x[-1]
        jac = np.zeros((n_pairs, 2 * N + 1))
        ci_i = ci[pairs[:, 0]]
        ci_j = ci[pairs[:, 1]]
        diffs = ci_i - ci_j  # (n_pairs, 2)
        # d/dx_i: 2 * diff  (for row i of ci)
        # d/dx_j: -2 * diff  (for row j of ci)
        # d/dr: -8*r
        for idx in range(n_pairs):
            i, j = pairs[idx]
            jac[idx, 2*i] = 2.0 * diffs[idx, 0]
            jac[idx, 2*i+1] = 2.0 * diffs[idx, 1]
            jac[idx, 2*j] = -2.0 * diffs[idx, 0]
            jac[idx, 2*j+1] = -2.0 * diffs[idx, 1]
            jac[idx, -1] = -8.0 * r
        return jac

    # Wall constraints: 0.5 - |c_ik| >= r  ->  0.5 - |c_ik| - r >= 0
    # Split into two per coordinate:
    #   For x-coord: if c_i0 >= 0: 0.5 - c_i0 - r >= 0; if c_i0 < 0: 0.5 + c_i0 - r >= 0
    # To keep it smooth, use: (0.5 - c_ik - r) for the "right/top" side
    # and (0.5 + c_ik - r) for the "left/bottom" side
    # Actually simpler: just use 4 wall constraints per circle (2 coords x 2 sides)
    # But that's 4N = 200 constraints. Let's use 2N: 0.5 - |c_ik| - r >= 0
    # This is non-smooth at 0. Use the smooth reformulation:
    #   min(0.5 - c_ik, 0.5 + c_ik) - r >= 0
    # Still non-smooth. Better: use 4N linear constraints.

    # Use 4N linear wall constraints:
    #   0.5 - c_i0 - r >= 0   (right wall)
    #   0.5 + c_i0 - r >= 0   (left wall)
    #   0.5 - c_i1 - r >= 0   (top wall)
    #   0.5 + c_i1 - r >= 0   (bottom wall)
    n_wall_constr = 4 * N

    def wall_constraints(x):
        ci = x[:2*N].reshape(N, 2)
        r = x[-1]
        # right: 0.5 - c_i0 - r
        # left:  0.5 + c_i0 - r
        # top:   0.5 - c_i1 - r
        # bottom:0.5 + c_i1 - r
        out = np.empty(n_wall_constr)
        out[0::4] = 0.5 - ci[:, 0] - r
        out[1::4] = 0.5 + ci[:, 0] - r
        out[2::4] = 0.5 - ci[:, 1] - r
        out[3::4] = 0.5 + ci[:, 1] - r
        return out

    def wall_constraints_jac(x):
        jac = np.zeros((n_wall_constr, 2 * N + 1))
        for i in range(N):
            base = 4 * i
            jac[base, 2*i] = -1.0      # 0.5 - c_i0 - r
            jac[base, -1] = -1.0
            jac[base+1, 2*i] = 1.0      # 0.5 + c_i0 - r
            jac[base+1, -1] = -1.0
            jac[base+2, 2*i+1] = -1.0   # 0.5 - c_i1 - r
            jac[base+2, -1] = -1.0
            jac[base+3, 2*i+1] = 1.0    # 0.5 + c_i1 - r
            jac[base+3, -1] = -1.0
        return jac

    # Combine all constraints
    def all_constraints(x):
        return np.concatenate([pair_constraints(x), wall_constraints(x)])

    def all_constraints_jac(x):
        return np.concatenate([pair_constraints_jac(x), wall_constraints_jac(x)], axis=1)

    constraints = [{'type': 'ineq', 'fun': all_constraints, 'jac': all_constraints_jac}]
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]

    # Run SLSQP with decreasing tolerance for precision
    slsqp_opts = [
        {'maxiter': 500, 'ftol': 1e-12, 'iprint': 0},
        {'maxiter': 300, 'ftol': 1e-14, 'iprint': 0},
    ]

    for opts in slsqp_opts:
        if (time.time() - t0) > t0 + time_budget:
            break
        try:
            res = minimize(
                objective,
                best_x,
                jac=objective_grad,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options=opts
            )
            if res.success or (res.fun < -best_r):
                r_new = -res.fun
                c_new = res.x[:2*N].reshape(N, 2)
                # Verify with exact radius
                r_exact = _radius_fast(c_new)
                if r_exact > best_r:
                    best_r = r_exact
                    best_x = np.concatenate([c_new.ravel(), [best_r]])
        except Exception:
            break

    return best_x[:2*N].reshape(N, 2), best_r


def _lbfgs_polish(c, t0, time_budget=10.0):
    """
    L-BFGS-B polish using soft-min radius with decreasing temperature.
    """
    x = c.ravel()
    best_x = x.copy()
    best_r = _radius_fast(c)

    temps = [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02]
    for t in temps:
        if (time.time() - t0) > t0 + time_budget:
            break
        res = minimize(
            lambda xx: _soft_min_radius_neg(xx, t),
            x,
            jac=lambda xx: _soft_min_radius_grad(xx, t),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 600, 'gtol': 1e-14, 'ftol': 1e-18}
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

    # Phase 2: SLSQP epigraph polish (direct optimization with hard constraints)
    if (time.time() - t0) < DEADLINE - 25:
        c_slsqp, r_slsqp = _slsqp_epigraph_polish(
            best_c, t0,
            time_budget=min(18.0, DEADLINE - 25 - (time.time() - t0))
        )
        if r_slsqp > best_r:
            best_r = r_slsqp
            best_c = c_slsqp

    # Phase 2b: L-BFGS-B soft-min polish as secondary
    if (time.time() - t0) < DEADLINE - 20:
        c_polished, r_polished = _lbfgs_polish(
            best_c, t0,
            time_budget=min(10.0, DEADLINE - 20 - (time.time() - t0))
        )
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished

    # Phase 3: Active-set refinement
    if (time.time() - t0) < DEADLINE - 15:
        c_qcqp, r_qcqp = _active_set_refine(best_c, best_r, max_iter=50)
        if r_qcqp > best_r:
            best_r = r_qcqp
            best_c = c_qcqp

    # Phase 4: Voronoi-style active-set refinement
    if (time.time() - t0) < DEADLINE - 10:
        c_voro, r_voro = _voronoi_style_refine(best_c, best_r, max_iter=30)
        if r_voro > best_r:
            best_r = r_voro
            best_c = c_voro

    # Phase 5: Second SLSQP epigraph pass with tighter tolerance
    if (time.time() - t0) < DEADLINE - 10:
        c_slsqp2, r_slsqp2 = _slsqp_epigraph_polish(
            best_c, t0,
            time_budget=min(8.0, DEADLINE - 10 - (time.time() - t0))
        )
        if r_slsqp2 > best_r:
            best_r = r_slsqp2
            best_c = c_slsqp2

    # Phase 6: Final high-precision local search
    if (time.time() - t0) < DEADLINE - 5:
        c_ls, r_ls = _final_local_search(best_c, best_r, max_iter=300, step_scale=1e-5)
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls

    # Phase 7: Ultra-fine SLSQP polish
    if (time.time() - t0) < DEADLINE - 2:
        c_ultra, r_ultra = _slsqp_epigraph_polish(
            best_c, t0,
            time_budget=min(3.0, DEADLINE - 2 - (time.time() - t0))
        )
        if r_ultra > best_r:
            best_r = r_ultra
            best_c = c_ultra

    # Final validation: ensure all points are strictly inside
    best_c = np.clip(best_c, -0.5 + 1e-9, 0.5 - 1e-9)

    return best_c
