"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from math import cos, sin, pi, sqrt

# EVOLVE-BLOCK-START
def construct():
    """
    Maximize the product of pairwise distances for 8 planar points with diameter 2.
    The regular octagon scores exactly 1.0. We use simulated annealing to find
    better configurations.
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave safety margin before 120s limit
    
    def compute_score(points):
        """Compute Delta/n^n where points are scaled to diameter 2."""
        n = len(points)
        # Compute all pairwise distances
        pts = np.array(points)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=-1))
        
        # Find diameter (max distance)
        diam = np.max(dists)
        if diam == 0:
            return 0.0
        
        # Scale to diameter 2
        scaled_dists = dists / diam * 2.0
        
        # Product of all ordered pairs (i != j)
        # Exclude diagonal (distance 0)
        mask = ~np.eye(n, dtype=bool)
        prod_dists = np.prod(scaled_dists[mask])
        
        # Score = prod_dists / n^n
        score = prod_dists / (n ** n)
        return score
    
    # Start with regular octagon
    regular_octagon = [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
    current_points = [p[:] for p in regular_octagon]
    current_score = compute_score(current_points)
    
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    # Simulated annealing
    n = 8
    T = 0.5  # Initial temperature
    cooling_rate = 0.999
    min_T = 1e-10
    
    # Precompute for speed
    def fast_score(points):
        """Faster score computation using numpy."""
        pts = np.array(points)
        # Pairwise distances
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diffs**2, axis=-1)
        dists = np.sqrt(dists_sq)
        
        diam = np.max(dists)
        if diam < 1e-15:
            return 0.0
        
        scaled_dists = dists / diam * 2.0
        mask = ~np.eye(n, dtype=bool)
        prod_dists = np.prod(scaled_dists[mask])
        return prod_dists / (n ** n)
    
    step_size = 0.1
    iterations_per_check = 1000
    
    for iteration in range(500000):
        if time.time() > deadline:
            break
        
        if iteration % iterations_per_check == 0:
            if T < min_T:
                break
        
        # Propose a move: perturb one point
        idx = random.randint(0, n - 1)
        new_points = [p[:] for p in current_points]
        
        # Random perturbation
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        new_points[idx][0] += dx
        new_points[idx][1] += dy
        
        new_score = fast_score(new_points)
        
        # Accept or reject
        if new_score >= current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        else:
            delta = new_score - current_score
            # Use log-space for acceptance probability to avoid underflow
            if T > 1e-15:
                accept_prob = min(1.0, np.exp(delta / T))
                if random.random() < accept_prob:
                    current_points = new_points
                    current_score = new_score
        
        # Cool down
        T *= cooling_rate
    
    # Return the best configuration found, scaled appropriately
    # The evaluator will scale to diameter 2, so we can return any scale
    # But let's return points with reasonable coordinates
    result = [[float(x), float(y)] for x, y in best_points]
    return result

# EVOLVE-BLOCK-END
