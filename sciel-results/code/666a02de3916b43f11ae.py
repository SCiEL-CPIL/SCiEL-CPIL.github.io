import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _compute_radius(centers):
    """Compute the exact largest common radius for given centers inside [-0.5, 0.5]^3."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    half = 0.5
    # Distance from each center to cube walls
    wall_dists = np.min(half - np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    return min(min_dist / 2.0, min_wall)

def _soft_min_objective(centers, beta=50.0):
    """
    Soft-min objective: minimizes a smooth approximation of the minimum sphere radius.
    r = min(min_pairwise_dist/2, min_wall_dist)
    We maximize r, so we minimize -r.
    Using log-sum-exp for soft-min:
    soft_min(x, beta) = -1/beta * log(sum(exp(-beta * x_i)))
    """
    centers = np.asarray(centers, dtype=np.float64)
    half = 0.5
    
    # Wall distances
    wall_dists = half - np.abs(centers)
    min_wall_per_sphere = np.min(wall_dists, axis=1)
    
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    # We want the minimum over all pairs i<j
    # Extract upper triangle
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    
    # Combine all "radius candidates"
    # Candidates: min_wall_per_sphere (N values), pair_dists (N*(N-1)/2 values)
    candidates = np.concatenate([min_wall_per_sphere, pair_dists])
    
    # Soft min
    # To avoid overflow in exp, subtract max
    c_max = np.max(candidates)
    log_sum = np.log(np.sum(np.exp(-beta * (candidates - c_max))))
    soft_min_val = c_max - (1.0 / beta) * log_sum
    
    # We want to maximize soft_min_val, so minimize -soft_min_val
    return -soft_min_val

def _build_initial_grid():
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.linspace(-0.375, 0.375, 4)
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid

def _optimize_from_initial(initial_centers, beta=50.0, max_iter=200, deadline=None):
    """
    Optimize from a given initial configuration using L-BFGS-B on the soft-min objective.
    """
    x0 = initial_centers.flatten()
    bounds = [(-0.5, 0.5) for _ in range(3 * N)]
    
    def obj(x_flat):
        c = x_flat.reshape(N, 3)
        return _soft_min_objective(c, beta)
    
    res = minimize(
        obj,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    if deadline is not None and time.time() > deadline:
        return initial_centers, _compute_radius(initial_centers)
        
    return res.x.reshape(N, 3), _compute_radius(res.x.reshape(N, 3))

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initial Grid Construction
    grid = _build_initial_grid()
    
    best_centers = grid
    best_r = _compute_radius(grid)
    
    # 2. Optimization Phase
    # We will use a few restarts with different initial guesses and betas.
    # Start with low beta (smooth) then increase (sharper).
    
    # Restart 1: Grid with low beta
    c1, r1 = _optimize_from_initial(grid, beta=20.0, max_iter=300, deadline=deadline)
    if r1 > best_r:
        best_r = r1
        best_centers = c1
    
    # Restart 2: Perturbed grid
    if time.time() < deadline - 20:
        noise = np.random.normal(0, 0.02, (N, 3))
        c_init2 = np.clip(grid + noise, -0.5, 0.5)
        c2, r2 = _optimize_from_initial(c_init2, beta=20.0, max_iter=300, deadline=deadline)
        if r2 > best_r:
            best_r = r2
            best_centers = c2

    # 3. Polish Phase: Increase beta for sharper minimum
    # Use the best result so far as initial guess
    if time.time() < deadline - 10:
        c3, r3 = _optimize_from_initial(best_centers, beta=100.0, max_iter=500, deadline=deadline)
        if r3 > best_r:
            best_r = r3
            best_centers = c3
        
        # One more polish with even higher beta
        if time.time() < deadline - 5:
            c4, r4 = _optimize_from_initial(best_centers, beta=500.0, max_iter=1000, deadline=deadline)
            if r4 > best_r:
                best_r = r4
                best_centers = c4

    return best_centers
