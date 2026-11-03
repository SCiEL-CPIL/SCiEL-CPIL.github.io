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


def _slsqp_epigraph_polish(c_init, r_init, t0, time_budget=30.0):
    """
    SLSQP polish using epigraph formulation.
    Variables: [x_1, y_1, ..., x_N, y_N, r]
    Objective: minimize -r
    Constraints:
      1. d_ij >= 2r for all i < j
      2. w_ik >= r for all i, k
    """
    x0 = np.concatenate([c_init.ravel(), [r_init]])
    
    # Precompute indices for constraints
    pair_indices = [(i, j) for i in range(N) for j in range(i + 1, N)]
    wall_indices = [(i, k) for i in range(N) for k in range(2)]
    n_pairs = len(pair_indices)
    n_walls = len(wall_indices)
    n_cons = n_pairs + n_walls

    def objective(x):
        return -x[-1]

    def objective_grad(x):
        g = np.zeros_like(x)
        g[-1] = -1.0
        return g

    def constraints_fun(x):
        c = x[:2 * N].reshape(N, 2)
        r = x[-1]
        cons = np.zeros(n_cons)
        
        # Pair constraints: d_ij - 2r >= 0
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        row = 0
        for (i, j) in pair_indices:
            cons[row] = d[i, j] - 2 * r
            row += 1
            
        # Wall constraints: w_ik - r >= 0
        wall_dist = 0.5 - np.abs(c)
        for (i, k) in wall_indices:
            cons[row] = wall_dist[i, k] - r
            row += 1
            
        return cons

    def constraints_jac(x):
        c = x[:2 * N].reshape(N, 2)
        r = x[-1]
        J = np.zeros((n_cons, 2 * N + 1))
        
        # Pair constraints
        d = cdist(c, c)
        np.fill_diagonal(d, 1.0)
        row = 0
        for (i, j) in pair_indices:
            dij = c[i] - c[j]
            norm = d[i, j]
            if norm > 1e-12:
                J[row, 2 * i] = dij[0] / norm
                J[row, 2 * i + 1] = dij[1] / norm
                J[row, 2 * j] = -dij[0] / norm
                J[row, 2 * j + 1] = -dij[1] / norm
            J[row, -1] = -2.0
            row += 1
            
        # Wall constraints
        for (i, k) in wall_indices:
            sign = np.sign(c[i, k])
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
            options={'maxiter': 1000, 'ftol': 1e-14, 'maxls': 50}
        )
        
        c_new = res.x[:2 * N].reshape(N, 2)
        r_check = _radius_fast(c_new)
        if r_check > r_init:
            return c_new, r_check
    except Exception:
        pass
        
    return c_init, r_init


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
        if (time.time() - t0) > DEADLINE - 25:
            break
        # Phase 1: hard-disk radius expansion
        c, r = _grow_radius(start, 0.04, 0.072, 0.0005, 600, 0.015, 0.001, t0)
        if r > best_r:
            best_r = r
            best_c = c.copy()
    
    if best_c is None:
        best_c = inits[0]
        best_r = _radius_fast(best_c)
    
    # Phase 2: L-BFGS-B soft-min polish
    if (time.time() - t0) < DEADLINE - 20:
        c_polished, r_polished = _lbfgs_polish(best_c, t0, time_budget=min(15.0, DEADLINE - 20 - (time.time() - t0)))
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished
    
    # Phase 3: SLSQP epigraph polish
    if (time.time() - t0) < DEADLINE - 5:
        c_slq, r_slq = _slsqp_epigraph_polish(best_c, best_r, t0, time_budget=min(30.0, DEADLINE - 5 - (time.time() - t0)))
        if r_slq > best_r:
            best_r = r_slq
            best_c = c_slq
    
    # Final validation: ensure all points are strictly inside
    best_c = np.clip(best_c, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_c
