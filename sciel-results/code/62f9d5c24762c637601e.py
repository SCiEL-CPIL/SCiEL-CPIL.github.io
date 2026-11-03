"""Improved program using Multi-Start L-BFGS-B with Soft-Min Objective and Nelder-Mead Polish."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _calculate_true_radius(centers):
    """Calculate the true maximum radius for a given set of centers."""
    # N=50 is small enough for O(N^2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def _soft_min_objective(params, k):
    """
    Objective function for L-BFGS-B.
    Maximizes the minimum distance using a log-sum-exp soft-min approximation.
    Returns negative soft-min to be minimized by L-BFGS-B.
    """
    centers = params.reshape(N, 2)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    pair_dists = np.sqrt(dists_sq) / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    
    # Combine all constraints
    all_dists = np.concatenate([pair_dists.flatten(), wall_dists.flatten()])
    
    # Filter out non-finite values (should not happen with valid inputs, but safe)
    all_dists = all_dists[np.isfinite(all_dists)]
    
    if len(all_dists) == 0:
        return 0.0
    
    # Soft min approximation: -1/k * log(sum(exp(-k * d)))
    # To avoid overflow/underflow, shift by min
    m = np.min(all_dists)
    shifted = all_dists - m
    exp_terms = np.exp(-k * shifted)
    log_sum = np.log(np.sum(exp_terms))
    soft_min = m - (1.0 / k) * log_sum
    
    return -soft_min

def _optimize_from_start(start_centers, bounds, time_limit=10.0):
    """
    Optimize a single starting configuration using L-BFGS-B followed by Nelder-Mead.
    """
    best_centers = start_centers.copy()
    best_score = _calculate_true_radius(best_centers)
    
    # Phase 1: L-BFGS-B with Soft-Min
    # Use a moderately high k for smoothness but enough to approximate min
    k = 500.0 
    try:
        result = minimize(
            _soft_min_objective, 
            best_centers.reshape(-1), 
            args=(k,),
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        centers = result.x.reshape(N, 2)
        true_r = _calculate_true_radius(centers)
        
        if true_r > best_score:
            best_score = true_r
            best_centers = centers
    except Exception:
        pass

    # Phase 2: Nelder-Mead Polish on the best candidate
    # Nelder-Mead is derivative-free and can polish the solution further
    # We use a simple objective: -min_dist
    def nm_objective(params):
        centers = params.reshape(N, 2)
        return -_calculate_true_radius(centers)

    try:
        result_nm = minimize(
            nm_objective,
            best_centers.reshape(-1),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        centers_nm = result_nm.x.reshape(N, 2)
        # Clip to ensure validity
        centers_nm = np.clip(centers_nm, -0.4999, 0.4999)
        true_r_nm = _calculate_true_radius(centers_nm)
        
        if true_r_nm > best_score:
            best_score = true_r_nm
            best_centers = centers_nm
    except Exception:
        pass

    return best_centers, best_score

def construct():
    bounds = [(-0.4999, 0.4999)] * (2 * N)
    
    inits = []
    
    # 1. Hexagonal grid (dense)
    s = 0.13
    points = []
    range_x = np.arange(-0.8, 0.8, s)
    range_y = np.arange(-0.8, 0.8, s)
    for i, x in enumerate(range_x):
        offset = (s / 2) if i % 2 == 0 else 0
        for y in range_y:
            points.append((x + offset, y))
    points = np.array(points)
    mask = (np.abs(points[:, 0]) <= 0.49) & (np.abs(points[:, 1]) <= 0.49)
    points = points[mask]
    if len(points) >= N:
        np.random.seed(42)
        indices = np.random.choice(len(points), N, replace=False)
        inits.append(points[indices])
        
    # 2. Square grid
    grid_x = np.linspace(-0.45, 0.45, 7)
    grid_y = np.linspace(-0.45, 0.45, 7)
    grid_points = np.array([[x, y] for x in grid_x for y in grid_y])
    if len(grid_points) >= N:
        np.random.seed(123)
        indices = np.random.choice(len(grid_points), N, replace=False)
        inits.append(grid_points[indices])
        
    # 3. Random uniform (multiple seeds)
    for seed in [999, 1000, 1001, 1002]:
        np.random.seed(seed)
        inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # 4. Jittered hex
    if len(points) >= N:
        np.random.seed(43)
        indices = np.random.choice(len(points), N, replace=False)
        jittered = points[indices] + np.random.normal(0, 0.03, (N, 2))
        jittered = np.clip(jittered, -0.49, 0.49)
        inits.append(jittered)
        
    # 5. Randomized Grid (Poisson-like)
    np.random.seed(555)
    # Generate a set of random points and keep the ones that are far enough apart
    # This is a simple rejection sampling for a good initial spread
    random_points = []
    for _ in range(200):
        p = np.random.uniform(-0.45, 0.45, 2)
        if all(np.linalg.norm(p - q) > 0.15 for q in random_points):
            random_points.append(p)
        if len(random_points) == N:
            break
    if len(random_points) == N:
        inits.append(np.array(random_points))

    best_centers = None
    best_score = -1.0
    
    start_time = time.time()
    
    # Process inits, prioritizing structured ones first, then random
    # We have ~8 inits. Each takes ~10-15s. Total ~80-120s.
    # To be safe, we limit total time to 110s.
    
    for i, init in enumerate(inits):
        if time.time() - start_time > 105.0: 
            break
            
        centers, score = _optimize_from_start(init, bounds, time_limit=12.0)
        if score > best_score:
            best_score = score
            best_centers = centers
            
    if best_centers is None:
        # Fallback
        best_centers = np.random.uniform(-0.45, 0.45, (N, 2))
        
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
