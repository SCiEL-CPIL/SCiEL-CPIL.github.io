"""
Solves the problem of packing 50 equal spheres in a unit cube.
Uses a smooth surrogate (soft-min) for the minimum distance objective,
optimized via L-BFGS-B with multiple random restarts.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 50
TIME_LIMIT = 115.0  # Safety margin for 120s limit
START_TIME = time.time()

def compute_min_radius(pos):
    """
    Compute the actual maximum radius for a given configuration.
    pos: (N, 3) array of centers.
    Returns: min(min_pairwise_dist/2, min_wall_dist/2)
    """
    # Pairwise distances
    # pos shape (N, 3)
    # diff shape (N, N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    # Zero out diagonal
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    
    # Distances to walls
    # pos in [-0.5, 0.5]
    # dist to wall is min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y, 0.5 - z, 0.5 + z)
    # Actually, distance to boundary of cube [-0.5, 0.5]^3
    # For a point (x,y,z), dist to wall is min(0.5-abs(x), 0.5-abs(y), 0.5-abs(z))
    abs_pos = np.abs(pos)
    wall_dists = 0.5 - abs_pos # (N, 3)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist / 2.0)
    return r

def objective(pos_flat, beta):
    """
    Objective function for minimization.
    Minimizes -soft_min(distances) to maximize the minimum distance.
    """
    pos = pos_flat.reshape(-1, 3)
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 0.0)
    
    # We need distances, not squared distances, for the log-sum-exp
    # But exp(-beta * sqrt(d^2)) is expensive. 
    # Approximation: use squared distances? No, the metric is Euclidean distance.
    # Let's compute distances.
    dists = np.sqrt(dist_sq)
    
    # Distances to walls
    abs_pos = np.abs(pos)
    wall_dists = 0.5 - abs_pos # (N, 3)
    
    # Combine all relevant distances
    # We want to maximize the minimum of all pair distances and wall distances.
    # Vector of all distances:
    # Upper triangle of dists
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # All wall distances
    wall_dists_flat = wall_dists.flatten()
    
    all_dists = np.concatenate([pair_dists, wall_dists_flat])
    
    # Soft min: -1/beta * log(sum(exp(-beta * dists)))
    # Numerical stability: shift by max
    # max_d = np.max(all_dists)
    # soft_min = -1.0/beta * (np.log(np.sum(np.exp(-beta * (all_dists - max_d)))) - beta * max_d)
    # This simplifies to:
    # soft_min = max_d - 1.0/beta * np.log(np.sum(np.exp(-beta * (all_dists - max_d))))
    
    # To avoid overflow in exp, we use the log-sum-exp trick.
    # Let M = max(all_dists)
    # sum(exp(-beta * d)) = exp(-beta * M) * sum(exp(-beta * (d - M)))
    # log(sum(...)) = -beta * M + log(sum(exp(-beta * (d - M))))
    # soft_min = -1/beta * (-beta * M + log(...)) = M - 1/beta * log(...)
    
    M = np.max(all_dists)
    # exp(-beta * (all_dists - M)) is safe because (all_dists - M) <= 0
    # If beta is large, exp(-large positive) -> 0.
    # The term (M - M) = 0, so exp(0)=1.
    terms = np.exp(-beta * (all_dists - M))
    log_sum = np.log(np.sum(terms))
    soft_min = M - (1.0 / beta) * log_sum
    
    return -soft_min

def construct():
    global START_TIME, TIME_LIMIT
    
    best_pos = None
    best_radius = -1.0
    
    # Number of restarts. We will break early if time runs out.
    # L-BFGS-B is fast for N=50.
    n_restarts = 10 
    beta = 50.0 # Smoothness parameter. Higher is more accurate but harder to optimize.
    
    for i in range(n_restarts):
        if time.time() - START_TIME > TIME_LIMIT - 5.0:
            break
            
        # Initialize positions
        # Random uniform in [-0.5, 0.5]
        # To avoid immediate overlaps, we can use a jittered grid for the first few
        if i == 0:
            # Jittered grid initialization
            m = 4 # 4x4x4 = 64 points
            grid = np.array([(x, y, z) for z in np.linspace(-0.4, 0.4, m) 
                             for y in np.linspace(-0.4, 0.4, m) 
                             for x in np.linspace(-0.4, 0.4, m)])
            # Shuffle and take first N
            np.random.shuffle(grid)
            pos0 = grid[:N].copy()
            # Add small jitter
            pos0 += np.random.uniform(-0.01, 0.01, pos0.shape)
        else:
            pos0 = np.random.uniform(-0.5, 0.5, (N, 3))
        
        pos0_flat = pos0.flatten()
        
        # Bounds
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(objective, pos0_flat, args=(beta,), 
                           method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < 0: # L-BFGS-B might not report success but improve
                pos_opt = res.x.reshape(-1, 3)
                # Ensure strict validity (clip to bounds just in case of floating point errors)
                pos_opt = np.clip(pos_opt, -0.5, 0.5)
                
                r = compute_min_radius(pos_opt)
                
                if r > best_radius:
                    best_radius = r
                    best_pos = pos_opt
                    
        except Exception as e:
            continue
            
    if best_pos is None:
        # Fallback: simple grid
        m = 4
        grid = np.array([(x, y, z) for z in np.linspace(-0.5, 0.5, m) 
                         for y in np.linspace(-0.5, 0.5, m) 
                         for x in np.linspace(-0.5, 0.5, m)])
        best_pos = grid[:N]
        
    return best_pos
