"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import cKDTree
import time
import random

N = 50

# EVOLVE-BLOCK-START

def _score(centers):
    """Compute the packing radius for a set of centers in [-0.5, 0.5]^3."""
    # Distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    tree = cKDTree(centers)
    pairs = tree.query_pairs(0.5)  # All pairs within 0.5 (all pairs for our scale)
    if len(pairs) == 0:
        return min_wall / 2.0
    
    min_pair_dist = float('inf')
    for i, j in pairs:
        d = np.linalg.norm(centers[i] - centers[j])
        if d < min_pair_dist:
            min_pair_dist = d
    
    return min(min_pair_dist, min_wall * 2) / 2.0

def _initial_grid():
    """Create an initial cubic grid arrangement."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _optimize(centers, time_limit, seed=42):
    """Optimize sphere centers using gradient ascent on the min-distance objective."""
    rng = np.random.default_rng(seed)
    best_centers = centers.copy()
    best_score = _score(centers)
    
    # Nesterov momentum parameters
    beta = 0.9
    step_size = 0.01
    patience = 50
    no_improve = 0
    
    start_time = time.time()
    prev_centers = centers.copy()
    
    for iteration in range(5000):
        if time.time() - start_time > time_limit - 2.0:
            break
        
        # Compute gradient of min distance
        tree = cKDTree(centers)
        pairs = tree.query_pairs(0.5)
        
        grad = np.zeros_like(centers)
        
        # Wall constraints
        for i in range(N):
            for k in range(3):
                if centers[i, k] > 0:
                    grad[i, k] -= 1.0
                else:
                    grad[i, k] += 1.0
        
        # Pair constraints - only consider closest pairs
        min_dist = float('inf')
        for i, j in pairs:
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                min_dist = d
        
        # Use soft-min approximation for gradient
        beta_lse = 200.0
        for i, j in pairs:
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 1e-10:
                continue
            weight = np.exp(-beta_lse * (d - min_dist))
            if weight > 1e-10:
                diff = centers[i] - centers[j]
                grad[i] += weight * diff / d
                grad[j] -= weight * diff / d
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
        
        # Nesterov update
        v = beta * (prev_centers - centers)
        prev_centers = centers.copy()
        centers_new = centers + step_size * (grad + v)
        
        # Keep centers within bounds
        centers_new = np.clip(centers_new, -0.49, 0.49)
        
        # Check improvement
        new_score = _score(centers_new)
        if new_score > best_score:
            best_score = new_score
            best_centers = centers_new.copy()
            no_improve = 0
            # Adaptive step size
            step_size = min(step_size * 1.05, 0.05)
        else:
            no_improve += 1
            step_size = max(step_size * 0.95, 0.001)
        
        centers = centers_new
        
        if no_improve > patience:
            break
    
    return best_centers, best_score

def construct():
    """Construct 50 sphere centers in the unit cube to maximize packing radius."""
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    # Start with grid
    centers = _initial_grid()
    best_centers = centers.copy()
    best_score = _score(centers)
    
    # Multiple restarts with different initializations
    attempts = [
        ("grid", centers),
        ("jittered_grid", centers + np.random.default_rng(1).normal(0, 0.02, centers.shape)),
        ("random", np.random.default_rng(2).uniform(-0.45, 0.45, (N, 3))),
    ]
    
    for name, init_centers in attempts:
        if time.time() - start_time > time_limit - 30.0:
            break
        
        time_remaining = time_limit - (time.time() - start_time) - 5.0
        if time_remaining < 5.0:
            break
        
        opt_centers, opt_score = _optimize(init_centers, time_remaining)
        
        if opt_score > best_score:
            best_score = opt_score
            best_centers = opt_centers
    
    # Final polish with tighter settings
    if time.time() - start_time < time_limit - 10.0:
        time_remaining = time_limit - (time.time() - start_time) - 2.0
        if time_remaining > 5.0:
            polished, polished_score = _optimize(best_centers, time_remaining, seed=123)
            if polished_score > best_score:
                best_score = polished_score
                best_centers = polished
    
    # Ensure valid output
    best_centers = np.clip(best_centers, -0.499999, 0.499999)
    
    return best_centers

# EVOLVE-BLOCK-END
