"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(centers):
    """
    Compute the valid radius for a given set of centers.
    Uses vectorized operations for speed.
    """
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to boundaries
    dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary_dist = np.min(dist_to_boundary)
    
    return min(min_pair_dist, min_boundary_dist) / 2.0

def _init_fcc(seed=42):
    """
    Initialize centers using a BCC lattice structure, which is a dense packing.
    """
    rng = np.random.RandomState(seed)
    
    points = []
    step = 1.0 / 3.0
    offset = -0.5
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                points.append([offset + i * step, offset + j * step, offset + k * step])
                points.append([offset + i * step + step/2, offset + j * step + step/2, offset + k * step + step/2])
    
    points = np.array(points)
    points = points[:N]
    points += rng.randn(N, 3) * 0.01
    points = np.clip(points, -0.499, 0.499)
    
    return points

def _simulated_annealing(centers, time_budget=10.0):
    """
    Simulated Annealing to maximize the minimum distance (radius).
    """
    start_time = time.time()
    current_centers = centers.copy()
    current_radius = _compute_radius(current_centers)
    best_centers = current_centers.copy()
    best_radius = current_radius
    
    T = 0.05
    alpha = 0.99
    min_T = 1e-5
    
    n = N
    
    while time.time() - start_time < time_budget:
        idx = np.random.randint(n)
        delta = np.random.randn(3) * T
        
        new_centers = current_centers.copy()
        new_centers[idx] += delta
        new_centers[idx] = np.clip(new_centers[idx], -0.499, 0.499)
        
        new_radius = _compute_radius(new_centers)
        
        delta_r = new_radius - current_radius
        if delta_r > 0:
            current_centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = current_centers.copy()
        else:
            if np.random.rand() < np.exp(-delta_r / T):
                current_centers = new_centers
                current_radius = new_radius
        
        T *= alpha
        if T < min_T:
            T = min_T * 2
    
    return best_centers

def construct():
    np.random.seed(42)
    
    start_time = time.time()
    time_limit = 110.0
    
    best_centers = None
    best_radius = 0.0
    
    init_strategies = []
    init_strategies.append(_init_fcc(42))
    
    for seed in range(10):
        rng = np.random.RandomState(seed)
        init_strategies.append(rng.uniform(-0.45, 0.45, (N, 3)))
    
    for i, init_centers in enumerate(init_strategies):
        if time.time() - start_time > time_limit:
            break
        
        remaining_time = time_limit - (time.time() - start_time)
        n_remaining = len(init_strategies) - i
        time_budget = min(remaining_time / n_remaining, 10.0)
        
        if time_budget < 1.0:
            break
        
        optimized = _simulated_annealing(init_centers, time_budget=time_budget)
        radius = _compute_radius(optimized)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = optimized.copy()
    
    if best_centers is not None and time.time() - start_time < time_limit - 5.0:
        remaining_time = time_limit - (time.time() - start_time)
        refined = _simulated_annealing(best_centers, time_budget=remaining_time * 0.8)
        refined_radius = _compute_radius(refined)
        if refined_radius >= best_radius:
            best_centers = refined
    
    return best_centers

# EVOLVE-BLOCK-END
