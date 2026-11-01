import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _score(centers):
    """
    Compute the maximum common radius for given centers.
    Returns the radius.
    """
    centers = np.asarray(centers)
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        return -np.inf
    
    # Wall constraints
    wall_min = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_min)
    
    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    
    return min(r_wall, r_pair)

def _soft_min(values, beta=50.0):
    """Smooth approximation to min via negative log-sum-exp."""
    values = np.asarray(values)
    m = np.min(values)
    exponent = -beta * (values - m)
    exponent = np.clip(exponent, -700, 700)
    log_sum = np.log(np.sum(np.exp(exponent)))
    return m - log_sum / beta

def _smooth_objective(x, beta=50.0):
    """
    Smooth surrogate: minimize -soft_min(all constraints) + boundary penalty.
    """
    c = x.reshape(-1, 3)
    
    # Wall constraints: 0.5 - |c_ik| for each coordinate
    wall_constraints = 0.5 - np.abs(c).ravel()
    
    # Pairwise half-distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_constraints = dists / 2.0
    pair_constraints = pair_constraints[np.triu_indices_from(pair_constraints, k=1)]
    
    # Combine all constraints
    all_constraints = np.concatenate([wall_constraints, pair_constraints])
    
    # Soft-min of all constraints (smooth approximation to the true min)
    s = _soft_min(all_constraints, beta)
    
    # Penalty for going outside bounds (soft)
    out_of_bounds = np.maximum(0, np.abs(c) - 0.4999)
    penalty = 10.0 * np.sum(out_of_bounds ** 2)
    
    return -s + penalty

def _initial_grid():
    """4x4x4 grid initialization, pick 50."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(grid))
    return grid[perm[:N]].copy()

def _initial_fcc_like():
    """FCC-like layered hexagonal initialization."""
    centers = np.zeros((N, 3))
    n_layers = 5
    per_layer = N // n_layers
    remaining = N - per_layer * n_layers
    
    for layer in range(n_layers):
        z = (layer + 0.5) / n_layers - 0.5
        n_in_layer = per_layer + (1 if layer < remaining else 0)
        
        for i in range(n_in_layer):
            row = i // 4
            col = i % 4
            x = (col + 0.5) / 4.0 - 0.5
            y = (row + 0.5) / 3.0 - 0.5
            if row % 2 == 1:
                x += 0.25 / 4.0
            idx = layer * per_layer + i
            if idx < N:
                centers[idx] = [x, y, z]
    
    # Fill any remaining
    idx = per_layer * n_layers
    while idx < N:
        centers[idx] = [0, 0, 0]
        idx += 1
    
    return centers

def _local_refine(centers, deadline, max_iters=5000):
    """
    Local refinement using subgradient method on the bottleneck constraints.
    Moves spheres to increase the minimum distance to walls or other spheres.
    """
    centers = centers.copy()
    best_score = _score(centers)
    step_size = 0.005
    no_improve_count = 0
    rng = np.random.RandomState(42)
    
    for _ in range(max_iters):
        if time.time() > deadline - 2:
            break
            
        # Compute current radius and identify bottlenecks
        # Wall distances
        wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
        min_wall_idx = np.argmin(wall_dists)
        min_wall_dist = wall_dists[min_wall_idx]
        
        # Pair distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        
        r_wall = min_wall_dist
        r_pair = min_pair_dist / 2.0
        
        current_r = min(r_wall, r_pair)
        
        # If current radius is worse than best, we might want to revert, 
        # but since we only accept improvements, we just check for improvement.
        
        improved = False
        new_centers = centers.copy()
        
        # Determine which constraint is active
        if r_wall < r_pair:
            # Wall is the bottleneck
            i = min_wall_idx
            # Move away from the wall
            for k in range(3):
                if centers[i, k] > 0:
                    new_centers[i, k] += step_size
                else:
                    new_centers[i, k] -= step_size
            # Clip to cube
            new_centers = np.clip(new_centers, -0.5, 0.5)
        else:
            # Pair is the bottleneck
            i, j = min_pair_idx
            # Move i away from j
            vec = centers[i] - centers[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                dir_vec = vec / norm
                new_centers[i] += step_size * dir_vec
                new_centers[j] -= step_size * dir_vec
            # Clip to cube
            new_centers = np.clip(new_centers, -0.5, 0.5)
        
        # Calculate new score
        new_score = _score(new_centers)
        
        if new_score > best_score + 1e-12:
            centers = new_centers
            best_score = new_score
            improved = True
            no_improve_count = 0
        else:
            no_improve_count += 1
            # If stuck, try a random jitter on the bottleneck sphere(s)
            if no_improve_count > 100:
                # Random perturbation
                if r_wall < r_pair:
                    i = min_wall_idx
                else:
                    i, j = min_pair_idx
                    # Perturb both
                    centers[i] += rng.normal(0, step_size, 3)
                    centers[j] += rng.normal(0, step_size, 3)
                centers[i] += rng.normal(0, step_size, 3)
                centers = np.clip(centers, -0.5, 0.5)
                no_improve_count = 0
                # Reduce step size if we've been stuck for a while
                if step_size > 0.0001:
                    step_size *= 0.9
            elif no_improve_count > 20:
                # Try smaller step
                step_size *= 0.95
                if step_size < 1e-6:
                    break

    return centers, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # Multiple initializations
    inits = [
        ("grid", _initial_grid()),
        ("fcc", _initial_fcc_like()),
    ]
    
    # Add a perturbed grid
    rng = np.random.RandomState(123)
    perturbed = _initial_grid() + rng.normal(0, 0.02, (N, 3))
    perturbed = np.clip(perturbed, -0.5, 0.5)
    inits.append(("perturbed", perturbed))
    
    best_centers = None
    best_score = -np.inf
    
    for name, init in inits:
        if time.time() > deadline - 20:
            break
        
        cur = init.copy()
        cur_score = _score(cur)
        
        if cur_score > best_score:
            best_score = cur_score
            best_centers = cur.copy()
        
        # Phase 1: Smooth surrogate optimization with annealed beta
        for beta in [20.0, 50.0, 100.0]:
            if time.time() > deadline - 10:
                break
            
            try:
                res = minimize(
                    _smooth_objective,
                    cur.ravel(),
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * 3 * N,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20}
                )
                
                new_centers = res.x.reshape(-1, 3)
                new_score = _score(new_centers)
                
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_centers.copy()
                    cur = new_centers.copy()
            except Exception:
                pass
        
        # Phase 2: Local refinement (Subgradient + Jitter)
        if time.time() < deadline - 5:
            polished, polished_score = _local_refine(cur, deadline)
            if polished_score > best_score:
                best_score = polished_score
                best_centers = polished

    if best_centers is None:
        return _initial_grid()
    return best_centers
