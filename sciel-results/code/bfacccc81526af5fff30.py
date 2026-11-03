"""Pack 50 equal circles in a unit square, maximize common radius.

Approach:
  1. Hard-disk radius expansion (push overlapping disks apart, grow target R).
  2. Soft-min L-BFGS-B polish with decreasing temperature (epigraph surrogate).
  3. QCQP active-set refinement on the kissing graph: identify active
     constraints, build the linear system for a displacement that keeps
     active constraints tight while increasing r, solve via least squares.
  4. Repeat 2-3 until no improvement.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # safety margin before 150s kill


def _radius(c):
    """Compute max common radius for centres c (N,2) in [-0.5,0.5]^2."""
    wall_r = np.min(0.5 - np.abs(c))
    diff = c[:, None, :] - c[None, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_pair_d = np.sqrt(d2.min())
    return min(wall_r, min_pair_d / 2.0)


def _radius_fast(c):
    """Slightly faster radius using squared distances."""
    wall_r = np.min(0.5 - np.abs(c))
    diff = c[:, None, :] - c[None, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_pair_d2 = d2.min()
    pair_r = np.sqrt(min_pair_d2) / 2.0
    return min(wall_r, pair_r)


def _soft_min_radius(c, t):
    """
    Soft-min of all pairwise half-distances and wall distances.
    soft_min = -t * log(sum(exp(-x_j / t)))
    As t -> 0+, approaches true min.
    """
    wall = 0.5 - np.abs(c)  # (N, 2)
    diff = c[:, None, :] - c[None, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    pair_terms = np.sqrt(d2) / 2.0  # (N, N)

    all_terms = np.concatenate([wall.ravel(), pair_terms.ravel()])
    m = all_terms.min()
    s = np.exp(-(all_terms - m) / t)
    return m - t * np.log(s.sum())


def _soft_min_radius_neg(x, t):
    c = x.reshape(N, 2)
    return -_soft_min_radius(c, t)


def _soft_min_radius_grad(x, t):
    """
    Gradient of -soft_min w.r.t. x (flattened c).
    d(-soft_min)/dx = (1/Z) * sum_j exp(-(x_j - m)/t) * dx_j/dx
    where Z = sum_j exp(-(x_j - m)/t).
    """
    c = x.reshape(N, 2)
    wall = 0.5 - np.abs(c)
    diff = c[:, None, :] - c[None, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    dist = np.sqrt(d2)
    np.fill_diagonal(dist, 1.0)
    pair_terms = dist / 2.0

    all_terms = np.concatenate([wall.ravel(), pair_terms.ravel()])
    m = all_terms.min()
    exps = np.exp(-(all_terms - m) / t)
    Z = exps.sum()

    grad = np.zeros_like(c)

    # Wall part: term = 0.5 - |c_ik|, d/dc_ik = -sign(c_ik)
    w_exps = exps[:2 * N] / Z
    w_grad = -np.sign(c).ravel()
    grad_flat = np.zeros(2 * N)
    grad_flat += w_exps * w_grad

    # Pair part: term_ij = ||c_i - c_j||/2
    # d/dc_i = (c_i - c_j) / (2 * ||c_i - c_j||)
    # d/dc_j = (c_j - c_i) / (2 * ||c_i - c_j||)
    p_exps = exps[2 * N:] / Z  # (N, N)
    # For each (i,j): weight = p_exps[i,j] + p_exps[j,i]
    # grad[i] += (w_ij + w_ji) * (c_i - c_j) / (2 * d_ij)
    # grad[j] -= (w_ij + w_ji) * (c_i - c_j) / (2 * d_ij)
    total_w = p_exps + p_exps.T  # (N, N), symmetric
    np.fill_diagonal(total_w, 0.0)
    # g_ij = (c_i - c_j) / (2 * d_ij) for i != j
    g = diff / (2.0 * dist[:, :, None])
    np.fill_diagonal(g, 0.0)
    # grad[i] = sum_j total_w[i,j] * g[i,j]
    grad += np.sum(total_w[:, :, None] * g, axis=1)

    return grad.ravel()


def _relax_hard_disk(c, R, iters, eta0, eta1):
    """
    Hard-disk relaxation: push overlapping circles apart and keep them
    inside the square. Vectorized.
    """
    c = c.copy()
    n = len(c)
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


def _active_set_qcqp(c, r_cur, tol=1e-4, max_steps=100):
    """
    QCQP active-set refinement on the kissing graph.
    
    At the optimum, the packing is determined by a "rigid" graph of touching
    circles. We identify the active constraints (pairs at distance ~2r_cur,
    circles touching walls), then solve for a displacement delta such that:
      - All active pair constraints remain tight: ||c_i + delta_i - c_j - delta_j|| = 2(r_cur + dr)
      - All active wall constraints remain tight: (c_i + delta_i)_k = +/- (0.5 - (r_cur + dr))
      - dr is maximized.
    
    Linearization: for pair (i,j) with unit direction u_ij = (c_i - c_j)/||c_i - c_j||:
      (delta_i - delta_j) . u_ij = 2 * dr
    For wall (i, k) with normal n (inward):
      delta_i . n = dr
    
    We solve: find (delta_1..delta_N, dr) maximizing dr subject to these
    linearized constraints. This is a linear program, but we use a simpler
    least-squares approach: minimize ||A x - b||^2 where x = [delta_1..delta_N, dr].
    Actually, we want to maximize dr. Let's set up:
      A x = 0 (homogeneous, since we want to maintain tightness)
      maximize x_last (dr)
    with ||x|| <= eps (small step).
    
    Simpler practical approach: solve the linear system for the active
    constraints to find a feasible direction, then scale it.
    """
    c = c.copy()
    n = len(c)
    
    for step in range(max_steps):
        # Identify active constraints
        diff = c[:, None, :] - c[None, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        d = np.sqrt(d2)
        
        tol_d = max(tol, 1e-5)
        # Active pairs: distance close to 2*r_cur
        active_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if d[i, j] < 2 * r_cur + tol_d:
                    active_pairs.append((i, j))
        
        # Active walls: circles close to a wall
        wall_dist = 0.5 - np.abs(c)
        active_walls = []  # (i, k, sign) where sign = +1 means c_ik near +0.5, -1 means near -0.5
        for i in range(n):
            for k in range(2):
                if wall_dist[i, k] < r_cur + tol_d:
                    sign = 1 if c[i, k] > 0 else -1
                    active_walls.append((i, k, sign))
        
        if len(active_pairs) == 0 and len(active_walls) == 0:
            break
        
        # Build linear system: A x = 0, maximize x[-1] = dr
        # x = [delta_1 (2), ..., delta_n (2), dr] -> size 2n+1
        # For pair (i,j): (delta_i - delta_j) . u_ij = 2*dr
        #   => u_ij . delta_i - u_ij . delta_j - 2*dr = 0
        # For wall (i,k,sign): sign * delta_i[k] - dr = 0
        #   (inward normal: if sign=+1, wall at +0.5, inward is -x, so -delta_i[0] = dr => delta_i[0] = -dr)
        #   Wait: wall at x=+0.5, circle center at 0.5 - r. To increase r, center moves to 0.5 - (r+dr) = (0.5-r) - dr.
        #   So delta_i[0] = -dr. Inward normal is -e_x. delta . (-e_x) = dr => -delta[0] = dr => delta[0] = -dr.
        #   For wall at x=-0.5: center at -0.5 + r, increases to -0.5 + r + dr, delta[0] = +dr. Inward normal +e_x.
        
        n_vars = 2 * n + 1
        A = []
        for (i, j) in active_pairs:
            u = (c[i] - c[j]) / d[i, j]
            row = np.zeros(n_vars)
            row[2 * i] = u[0]
            row[2 * i + 1] = u[1]
            row[2 * j] = -u[0]
            row[2 * j + 1] = -u[1]
            row[-1] = -2.0
            A.append(row)
        
        for (i, k, sign) in active_walls:
            row = np.zeros(n_vars)
            # delta_i[k] = -sign * dr  (from above analysis)
            # sign * delta_i[k] + dr = 0  ... let me re-derive
            # If wall at +0.5 (sign=+1): center_x = 0.5 - r, new center_x = 0.5 - r - dr, delta_x = -dr
            #   => delta_x + dr = 0 => row[2i+0] = 1, row[-1] = 1
            # If wall at -0.5 (sign=-1): center_x = -0.5 + r, new center_x = -0.5 + r + dr, delta_x = +dr
            #   => delta_x - dr = 0 => row[2i+0] = 1, row[-1] = -1
            # General: sign * delta_i[k] + dr = 0? 
            #   sign=+1: delta + dr = 0 -> delta = -dr. Correct.
            #   sign=-1: -delta + dr = 0 -> delta = dr. Correct.
            row[2 * i + k] = sign
            row[-1] = 1.0
            A.append(row)
        
        if len(A) == 0:
            break
        
        A = np.array(A)  # (m, 2n+1)
        
        # Solve: find x in null(A) that maximizes x[-1], with ||x|| = 1.
        # This is: max x[-1] s.t. A x = 0, ||x|| = 1.
        # Equivalent to: find the component of e_last in the null space of A.
        # x = P e_last / ||P e_last|| where P projects onto null(A).
        # P = I - A^T (A A^T)^{-1} A
        
        try:
            # Project e_last onto null space of A
            e_last = np.zeros(n_vars)
            e_last[-1] = 1.0
            # P = I - A^T A^{-1} A  (for full row rank A)
            # Use SVD for robustness
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            # Null space basis: Vt rows corresponding to zero singular values
            # But A is m x n_vars, S has length min(m, n_vars)
            # Full Vt is n_vars x n_vars
            U_full, S_full, Vt_full = np.linalg.svd(A, full_matrices=True)
            # Null space dimension = n_vars - rank(A)
            rank = np.sum(S_full > 1e-10)
            null_basis = Vt_full[rank:]  # (n_vars - rank, n_vars)
            if len(null_basis) == 0:
                break
            # Project e_last onto null space
            coords = null_basis @ e_last  # (n_vars - rank,)
            norm_coords = np.linalg.norm(coords)
            if norm_coords < 1e-12:
                break
            x_dir = null_basis.T @ (coords / norm_coords)  # unit vector in null space with max dr component
            # x_dir[-1] should be > 0 (maximizing dr)
            if x_dir[-1] <= 0:
                x_dir = -x_dir
            if x_dir[-1] <= 0:
                break
            
            # Take a small step
            dr = x_dir[-1]
            # Scale so that the step is small
            delta = x_dir[:2 * n].reshape(n, 2)
            # Limit step size
            max_step = 1e-3
            step_scale = min(1.0, max_step / max(np.linalg.norm(delta), 1e-12))
            c_new = c + step_scale * delta
            r_new = _radius_fast(c_new)
            if r_new > r_cur + 1e-12:
                c = c_new
                r_cur = r_new
            else:
                break
        except np.linalg.LinAlgError:
            break
    
    return c, r_cur


def _init_7x7_plus():
    """7x7 grid (49 points) + 50th at centre, then jitter to remove duplicate."""
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
    """Hexagonal close-packed initial guess, clipped to square."""
    # Hexagonal lattice with spacing ~ 2*r ~ 0.14
    spacing = 0.14
    pts = []
    row = 0
    while len(pts) < N + 20:
        y = (row * spacing * np.sqrt(3) / 2) - 0.5
        x_offset = (spacing / 2) if row % 2 else 0.0
        col = 0
        while True:
            x = (col * spacing) - 0.5 + x_offset
            if x > 0.5:
                break
            pts.append((x, y))
            col += 1
        row += 1
    pts = np.array(pts[:N])
    # Center in square
    pts += 0.0
    c = pts + rng.uniform(-0.005, 0.005, (N, 2))
    c = np.clip(c, -0.48, 0.48)
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
        if (time.time() - t0) > DEADLINE - 15:
            break
    return best_c, best_r


def _lbfgs_polish(c, t0, time_budget=10.0):
    """
    L-BFGS-B polish using soft-min radius with decreasing temperature.
    """
    x = c.ravel()
    best_x = x.copy()
    best_r = _radius_fast(c)
    
    temps = [100.0, 50.0, 20.0, 10.0, 5.0, 2.0]
    for t in temps:
        if (time.time() - t0) > t0 + time_budget:
            break
        res = minimize(
            lambda xx: _soft_min_radius_neg(xx, t),
            x,
            jac=lambda xx: _soft_min_radius_grad(xx, t),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 300, 'gtol': 1e-9, 'ftol': 1e-12}
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
        if (time.time() - t0) > DEADLINE - 20:
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
    if (time.time() - t0) < DEADLINE - 15:
        time_budget = min(20.0, DEADLINE - 15 - (time.time() - t0))
        c_polished, r_polished = _lbfgs_polish(best_c, t0, time_budget=time_budget)
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished
    
    # Phase 3: Active-set QCQP refinement on kissing graph
    if (time.time() - t0) < DEADLINE - 5:
        c_as, r_as = _active_set_qcqp(best_c, best_r, tol=1e-4, max_steps=100)
        if r_as > best_r:
            best_r = r_as
            best_c = c_as
    
    # Phase 4: Final L-BFGS-B polish with very low temperature
    if (time.time() - t0) < DEADLINE - 2:
        x = best_c.ravel()
        res = minimize(
            lambda xx: _soft_min_radius_neg(xx, 1.0),
            x,
            jac=lambda xx: _soft_min_radius_grad(xx, 1.0),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-13}
        )
        c_final = res.x.reshape(N, 2)
        r_final = _radius_fast(c_final)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final
    
    # Phase 5: Another active-set pass
    if (time.time() - t0) < DEADLINE - 1:
        c_as2, r_as2 = _active_set_qcqp(best_c, best_r, tol=1e-5, max_steps=50)
        if r_as2 > best_r:
            best_r = r_as2
            best_c = c_as2
    
    # Final validation: ensure all points are strictly inside
    best_c = np.clip(best_c, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_c
