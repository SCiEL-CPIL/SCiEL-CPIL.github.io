"""Pack 50 equal circles in a unit square, maximize common radius.

Combines hard-disk radius expansion with a multi-temperature L-BFGS-B
soft-min polish and an SLSQP epigraph formulation for the final polish.
The soft-min objective is a smooth surrogate for the nonsmooth min-radius
function; as temperature t -> 0 it concentrates on the tightest constraints.
The SLSQP epigraph directly maximizes r subject to all distance and wall
constraints, providing a high-precision local optimum.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 118.0  # safety margin before 150s kill


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
    and wall distances. As t -> 0 this approaches the true min radius.
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

    grad_flat = np.zeros(2 * N)

    # Wall gradients: term = 0.5 - |c_ik|, d/dc_ik = -sign(c_ik)
    sign_c = np.sign(c)
    w_grad = -sign_c.ravel()
    grad_flat[:2 * N] += (exps[:2 * N] / Z) * w_grad

    # Pair gradients: vectorized computation
    # For each pair (i,j), weight = (exp(-pair_ij/t) + exp(-pair_ji/t)) / Z
    # Gradient contribution to c_i: weight * (c_i - c_j) / (2 * ||c_i - c_j||)
    # Gradient contribution to c_j: -weight * (c_i - c_j) / (2 * ||c_i - c_j||)
    
    # exps for pair terms
    pair_exps = exps[2 * N:]  # shape (N*N,)
    pair_exps = pair_exps.reshape(N, N)
    # Symmetrize: total weight for pair (i,j)
    pair_weights = (pair_exps + pair_exps.T) / Z  # shape (N, N)
    np.fill_diagonal(pair_weights, 0.0)
    
    # dist matrix for normalization
    dist = d.copy()
    np.fill_diagonal(dist, 1.0)
    safe_dist = np.where(dist > 1e-12, dist, 1.0)
    
    # diff[i,j] = c[i] - c[j], shape (N, N, 2)
    diff = c[:, None, :] - c[None, :, :]  # (N, N, 2)
    # g[i,j] = diff[i,j] / (2 * dist[i,j])
    g = diff / (2.0 * safe_dist[:, :, None])
    # Zero out diagonal
    idx = np.arange(N)
    g[idx, idx, :] = 0.0
    
    # grad_flat[i] gets sum_j pair_weights[i,j] * g[i,j]
    # grad_flat[j] gets -sum_i pair_weights[i,j] * g[i,j] = sum_i pair_weights[j,i] * g[j,i]
    # Since pair_weights is symmetric and g[j,i] = -g[i,j]:
    # grad for c_i = sum_j pair_weights[i,j] * g[i,j]
    grad_pair = np.sum(pair_weights[:, :, None] * g, axis=1)  # shape (N, 2)
    grad_flat[:2 * N] += grad_pair.ravel()

    return grad_flat


def _relax_hard_disk(c, R, iters, eta0, eta1):
    """
    Hard-disk relaxation: push overlapping circles apart and keep them
    inside the square. Vectorized.
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


def _active_set_polish(c_init, r_init, t0, time_budget=25.0):
    """
    Active-set polish: identify tight constraints, optimize only those.
    Much faster than full epigraph when few constraints are active.
    """
    c = c_init.copy()
    r = r_init
    best_c = c.copy()
    best_r = r
    
    # Identify active constraints (within tolerance of r)
    tol = 1e-6
    
    for iteration in range(5):
        if (time.time() - t0) > t0 + time_budget:
            break
        
        # Find active constraints
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        wall_dist = 0.5 - np.abs(c)
        
        # Active pairs: d_ij <= 2r + tol
        active_pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] <= 2 * r + tol:
                    active_pairs.append((i, j))
        
        # Active walls: wall_dist[i,k] <= r + tol
        active_walls = []
        for i in range(N):
            for k in range(2):
                if wall_dist[i, k] <= r + tol:
                    active_walls.append((i, k))
        
        n_active_pairs = len(active_pairs)
        n_active_walls = len(active_walls)
        n_cons = n_active_pairs + n_active_walls
        
        if n_cons == 0:
            break
        
        x0 = np.concatenate([c.ravel(), [r]])
        
        def objective(x):
            return -x[-1]

        def objective_grad(x):
            g = np.zeros_like(x)
            g[-1] = -1.0
            return g

        def constraints_fun(x):
            cc = x[:2 * N].reshape(N, 2)
            rr = x[-1]
            cons = np.zeros(n_cons)
            row = 0
            for (i, j) in active_pairs:
                dij = cc[i] - cc[j]
                cons[row] = np.sqrt(np.sum(dij**2)) - 2 * rr
                row += 1
            for (i, k) in active_walls:
                cons[row] = 0.5 - np.abs(cc[i, k]) - rr
                row += 1
            return cons

        def constraints_jac(x):
            cc = x[:2 * N].reshape(N, 2)
            J = np.zeros((n_cons, 2 * N + 1))
            row = 0
            for (i, j) in active_pairs:
                dij = cc[i] - cc[j]
                norm = np.sqrt(np.sum(dij**2))
                if norm > 1e-12:
                    J[row, 2 * i] = dij[0] / norm
                    J[row, 2 * i + 1] = dij[1] / norm
                    J[row, 2 * j] = -dij[0] / norm
                    J[row, 2 * j + 1] = -dij[1] / norm
                J[row, -1] = -2.0
                row += 1
            for (i, k) in active_walls:
                sign = np.sign(cc[i, k])
                if sign == 0:
                    sign = 1.0
                J[row, 2 * i + k] = -sign
                J[row, -1] = -1.0
                row += 1
            return J

        bounds = [(-0.5, 0.5)] * (2 * N) + [(0, 0.5)]
        cons = [{'type': 'ineq', 'fun': constraints_fun, 'jac': constraints_jac}]

        try:
            res = minimize(
                objective,
                x0,
                jac=objective_grad,
                bounds=bounds,
                constraints=cons,
                method='SLSQP',
                options={'maxiter': 500, 'ftol': 1e-14, 'maxls': 30}
            )
            
            c_new = res.x[:2 * N].reshape(N, 2)
            r_new = _radius_fast(c_new)
            if r_new > best_r + 1e-12:
                best_r = r_new
                best_c = c_new.copy()
                c = c_new.copy()
                r = r_new
            else:
                break
        except Exception:
            break
    
    return best_c, best_r


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
    """Hexagonal lattice initialization."""
    c = np.zeros((N, 2))
    # Hexagonal packing in a square
    dx = 1.0 / 7
    dy = dx * np.sqrt(3) / 2
    row = 0
    col = 0
    count = 0
    while count < N:
        x = -0.5 + dx / 2 + col * dx
        y = -0.5 + dy / 2 + row * dy
        if row % 2 == 1:
            x += dx / 2
        if -0.5 < x < 0.5 and -0.5 < y < 0.5:
            c[count] = [x, y]
            count += 1
        col += 1
        if col > 8:
            col = 0
            row += 1
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
        if (time.time() - t0) > DEADLINE - 15:
            break
    return best_c, best_r


def _lbfgs_polish(c, t0, time_budget=15.0):
    """
    L-BFGS-B polish using soft-min radius with decreasing temperature.
    """
    x = c.ravel()
    best_x = x.copy()
    best_r = _radius_fast(c)
    
    # Try a few temperature values, decreasing
    temps = [50.0, 20.0, 10.0, 5.0, 2.0]
    for t in temps:
        if (time.time() - t0) > t0 + time_budget:
            break
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
    
    return best_x.reshape(N, 2), best_r


def _langevin_optimize(c_init, t0, time_budget=10.0):
    """
    Langevin dynamics to explore the landscape and escape local minima.
    Uses the soft-min radius gradient.
    """
    x = c_init.ravel().copy()
    best_x = x.copy()
    best_r = _radius_fast(c_init)
    
    # Parameters
    eta = 0.005
    sigma = 0.005
    t_temp = 50.0 # Temperature for soft-min
    
    # Estimate steps
    start_time = time.time()
    max_steps = 5000
    
    for i in range(max_steps):
        if (time.time() - t0) > t0 + time_budget:
            break
            
        # Compute gradient
        try:
            grad = _soft_min_radius_grad(x, t_temp)
        except Exception:
            break
        
        # Update
        noise = np.random.normal(0, sigma, size=x.shape)
        x = x - eta * grad + noise * np.sqrt(eta)
        
        # Clip to bounds
        x = np.clip(x, -0.5, 0.5)
        
        # Evaluate
        c = x.reshape(N, 2)
        r = _radius_fast(c)
        
        if r > best_r:
            best_r = r
            best_x = x.copy()
            
        # Decay temperature and noise
        t_temp *= 0.999
        sigma *= 0.999
        eta *= 0.999
        
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
        c, r = _grow_radius(start, 0.04, 0.072, 0.0005, 600, 0.015, 0.001, t0)
        if r > best_r:
            best_r = r
            best_c = c.copy()
    
    if best_c is None:
        best_c = inits[0]
        best_r = _radius_fast(best_c)
    
    # Phase 2: Langevin Dynamics for exploration
    if (time.time() - t0) < DEADLINE - 20:
        budget = min(10.0, DEADLINE - 20 - (time.time() - t0))
        if budget > 2.0:
            c_lang, r_lang = _langevin_optimize(best_c, t0, time_budget=budget)
            if r_lang > best_r:
                best_r = r_lang
                best_c = c_lang
    
    # Phase 3: L-BFGS-B soft-min polish (vectorized gradient is faster now)
    if (time.time() - t0) < DEADLINE - 10:
        budget = min(10.0, DEADLINE - 10 - (time.time() - t0))
        if budget > 2.0:
            c_polished, r_polished = _lbfgs_polish(best_c, t0, time_budget=budget)
            if r_polished > best_r:
                best_r = r_polished
                best_c = c_polished
    
    # Phase 4: Active-set polish (much faster than full epigraph)
    if (time.time() - t0) < DEADLINE - 3:
        budget = min(15.0, DEADLINE - 3 - (time.time() - t0))
        if budget > 2.0:
            c_slq, r_slq = _active_set_polish(best_c, best_r, t0, time_budget=budget)
            if r_slq > best_r:
                best_r = r_slq
                best_c = c_slq
    
    # Final validation: ensure all points are strictly inside
    best_c = np.clip(best_c, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_c
