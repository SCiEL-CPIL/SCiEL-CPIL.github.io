"""
Structural change: Use scipy.optimize.minimize with L-BFGS-B on a smooth
soft-min objective, combined with multi-start from diverse initializations
and a final local polish using Nelder-Mead on the worst-constraint subset.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100
TIME_LIMIT = 140.0  # Safety margin before 150s kill


def _compute_radius(centres):
    """Compute the maximum radius for given centres."""
    n = len(centres)
    if n < 2:
        return (0.5 - np.abs(centres)).min()
    dist = cdist(centres, centres)
    np.fill_diagonal(dist, np.inf)
    r_pair = dist.min() / 2.0
    r_boundary = (0.5 - np.abs(centres)).min()
    return min(r_pair, r_boundary)


def _soft_min_objective(x_flat, temp=0.01):
    """
    Objective: -soft_min of all pairwise and boundary distances.
    We minimize this (i.e., maximize soft_min).
    """
    c = x_flat.reshape(-1, 3)
    n = c.shape[0]
    
    # Pairwise distances (upper triangle)
    dist = cdist(c, c)
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(n, k=1)
    pair_dists = dist[iu]
    
    # Boundary distances: min over 6 walls per point
    walls = np.stack([
        0.5 - c[:, 0], 0.5 + c[:, 0],
        0.5 - c[:, 1], 0.5 + c[:, 1],
        0.5 - c[:, 2], 0.5 + c[:, 2]
    ], axis=1)
    boundary_per_point = walls.min(axis=1)
    
    all_dists = np.concatenate([pair_dists, boundary_per_point])
    m = len(all_dists)
    
    # Numerically stable soft-min
    min_d = all_dists.min()
    shifted = all_dists - min_d
    weights = np.exp(-shifted / temp)
    w_sum = weights.sum()
    soft_min = min_d + temp * np.log(w_sum)
    
    return -soft_min


def _soft_min_grad(x_flat, temp=0.01):
    """Gradient of the soft-min objective."""
    c = x_flat.reshape(-1, 3)
    n = c.shape[0]
    
    dist = cdist(c, c)
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(n, k=1)
    pair_dists = dist[iu]
    
    walls = np.stack([
        0.5 - c[:, 0], 0.5 + c[:, 0],
        0.5 - c[:, 1], 0.5 + c[:, 1],
        0.5 - c[:, 2], 0.5 + c[:, 2]
    ], axis=1)
    boundary_per_point = walls.min(axis=1)
    
    all_dists = np.concatenate([pair_dists, boundary_per_point])
    min_d = all_dists.min()
    shifted = all_dists - min_d
    weights = np.exp(-shifted / temp)
    weights /= weights.sum()
    
    grad = np.zeros_like(c)
    
    # Pairwise contributions
    i_idx, j_idx = iu
    pair_weights = weights[:len(pair_dists)]
    
    for idx in range(len(pair_dists)):
        i, j = i_idx[idx], j_idx[idx]
        d = pair_dists[idx]
        if d > 1e-12:
            dir_vec = c[i] - c[j]
            w = pair_weights[idx]
            grad[i] += w * dir_vec / d
            grad[j] -= w * dir_vec / d
    
    # Boundary contributions
    bound_weights = weights[len(pair_dists):]
    for i in range(n):
        w = bound_weights[i]
        for k in range(3):
            d_left = 0.5 - c[i, k]
            d_right = 0.5 + c[i, k]
            if d_left < d_right:
                grad[i, k] += w
            else:
                grad[i, k] -= w
    
    # Objective is -soft_min, so gradient is -grad(soft_min)
    return -grad.flatten()


def _optimize_from_start(x0, time_budget=30.0, temp_start=0.02, temp_end=0.002):
    """
    Run L-BFGS-B optimization with annealed temperature.
    """
    start_time = time.time()
    n = x0.shape[0] // 3
    x = x0.copy()
    best_x = x.copy()
    best_r = _compute_radius(x.reshape(n, 3))
    
    # Multiple rounds with decreasing temperature
    n_rounds = 3
    for round_i in range(n_rounds):
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 5:
            break
        
        temp = temp_start * (temp_end / temp_start) ** (round_i / max(n_rounds - 1, 1))
        
        try:
            result = minimize(
                _soft_min_objective,
                x,
                jac=_soft_min_grad,
                args=(temp,),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-9, 0.5 - 1e-9)] * (n * 3),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            x = result.x
            r = _compute_radius(x.reshape(n, 3))
            if r > best_r:
                best_r = r
                best_x = x.copy()
        except Exception:
            pass
    
    return best_x.reshape(n, 3), best_r


def _nelder_mead_polish(centres, time_budget=10.0):
    """
    Polish using Nelder-Mead on a subset of points with worst constraints.
    """
    start_time = time.time()
    n = centres.shape[0]
    c = centres.copy()
    best_r = _compute_radius(c)
    
    # Identify worst pairs and boundary violations
    dist = cdist(c, c)
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(n, k=1)
    pair_dists = dist[iu]
    
    walls = np.stack([
        0.5 - c[:, 0], 0.5 + c[:, 0],
        0.5 - c[:, 1], 0.5 + c[:, 1],
        0.5 - c[:, 2], 0.5 + c[:, 2]
    ], axis=1)
    boundary_per_point = walls.min(axis=1)
    
    # Find points involved in the worst constraints
    worst_pairs_idx = np.argsort(pair_dists)[:20]
    worst_bound_idx = np.argsort(boundary_per_point)[:10]
    
    candidate_indices = set()
    for idx in worst_pairs_idx:
        i, j = iu[0][idx], iu[1][idx]
        candidate_indices.add(i)
        candidate_indices.add(j)
    for idx in worst_bound_idx:
        candidate_indices.add(idx)
    
    candidate_indices = sorted(candidate_indices)
    
    # Optimize each candidate point individually
    for pt_idx in candidate_indices:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        # Define objective for single point
        def obj_single(x_pt):
            c_new = c.copy()
            c_new[pt_idx] = x_pt
            r = _compute_radius(c_new)
            return -r
        
        x0 = c[pt_idx].copy()
        try:
            result = minimize(
                obj_single,
                x0,
                method='Nelder-Mead',
                bounds=[(-0.5 + 1e-9, 0.5 - 1e-9)] * 3,
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result.fun < -best_r:
                c[pt_idx] = result.x
                best_r = -result.fun
        except Exception:
            pass
    
    return c, best_r


def _initial_grid():
    """Create initial cubic grid packing 5x5x4 = 100."""
    m1, m2, m3 = 5, 5, 4
    xs = (np.arange(m1) + 0.5) / m1 - 0.5
    ys = (np.arange(m2) + 0.5) / m2 - 0.5
    zs = (np.arange(m3) + 0.5) / m3 - 0.5
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid.copy()


def _initial_fcc():
    """Create FCC-like initial configuration with perturbations."""
    m1, m2, m3 = 5, 5, 4
    xs = (np.arange(m1) + 0.5) / m1 - 0.5
    ys = (np.arange(m2) + 0.5) / m2 - 0.5
    zs = (np.arange(m3) + 0.5) / m3 - 0.5
    points = []
    for z in zs:
        for y in ys:
            for x in xs:
                points.append([x, y, z])
    rng = np.random.RandomState(123)
    points = np.array(points)
    points += rng.normal(0, 0.01, size=points.shape)
    points = np.clip(points, -0.49, 0.49)
    return points


def _random_initialization(rng, n=100):
    """Random initialization inside the cube."""
    points = rng.uniform(-0.45, 0.45, size=(n, 3))
    return points


def _bcs_initialization():
    """Body-centered cubic-like initialization."""
    # BCC: 8 corners + 1 center per sub-cube
    # For 100 points, use a 5x5x4 grid with BCC offsets
    m1, m2, m3 = 5, 5, 4
    xs = (np.arange(m1) + 0.5) / m1 - 0.5
    ys = (np.arange(m2) + 0.5) / m2 - 0.5
    zs = (np.arange(m3) + 0.5) / m3 - 0.5
    points = []
    for z in zs:
        for y in ys:
            for x in xs:
                points.append([x, y, z])
    # Add small BCC-like offsets
    rng = np.random.RandomState(456)
    points = np.array(points)
    offsets = rng.choice([[-0.02, -0.02, -0.02], [0.02, 0.02, 0.02]], 
                         size=(len(points), 1))
    points += offsets
    points = np.clip(points, -0.49, 0.49)
    return points


def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    # Generate multiple initial configurations
    inits = []
    inits.append(_initial_grid())
    inits.append(_initial_fcc())
    inits.append(_bcs_initialization())
    
    # Add a few random initializations
    for _ in range(2):
        inits.append(_random_initialization(rng))
    
    best_centres = inits[0]
    best_radius = _compute_radius(best_centres)
    
    # Allocate time budget across initializations
    elapsed = time.time() - start_time
    time_per_init = min(20.0, (TIME_LIMIT - elapsed - 15) / len(inits))
    
    for i, init in enumerate(inits):
        elapsed = time.time() - start_time
        remaining = TIME_LIMIT - elapsed - 10
        if remaining < 5:
            break
        
        t_budget = min(time_per_init, remaining - 5)
        if t_budget < 3:
            break
        
        # Run L-BFGS-B optimization
        try:
            result, r = _optimize_from_start(
                init.flatten(),
                time_budget=t_budget,
                temp_start=0.02,
                temp_end=0.002
            )
            if r > best_radius:
                best_radius = r
                best_centres = result
        except Exception:
            pass
    
    # Final polish with Nelder-Mead
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - elapsed - 2
    if remaining > 5:
        try:
            polished, r = _nelder_mead_polish(
                best_centres,
                time_budget=min(10.0, remaining - 2)
            )
            if r > best_radius:
                best_radius = r
                best_centres = polished
        except Exception:
            pass
    
    # Ensure validity
    best_centres = np.clip(best_centres, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return best_centres
