"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Place 50 equal non-overlapping circles in the unit square [-0.5, 0.5]^2
    to maximize the common radius.
    
    Uses Simulated Annealing followed by Hill Climbing.
    """
    
    def score(points):
        """Calculate the maximum radius for a given configuration."""
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        dist_x = 0.5 - np.abs(points[:, 0])
        dist_y = 0.5 - np.abs(points[:, 1])
        min_boundary_dist = np.min(np.minimum(dist_x, dist_y))
        
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        return r

    def grid_init():
        """Initialize points on a grid."""
        # 10x5 grid
        xs = np.linspace(-0.45, 0.45, 10)
        ys = np.linspace(-0.45, 0.45, 5)
        X, Y = np.meshgrid(xs, ys)
        points = np.column_stack((X.ravel(), Y.ravel()))
        return points

    np.random.seed(42)
    points = grid_init()
    current_score = score(points)
    best_points = points.copy()
    best_score = current_score
    
    start_time = time.time()
    time_limit = 110.0
    
    # Phase 1: Simulated Annealing
    T = 0.05
    T_end = 0.001
    alpha = 0.999
    step_size = 0.05
    
    while T > T_end:
        if time.time() - start_time > time_limit:
            break
            
        idx = np.random.randint(N)
        delta = np.random.normal(0, step_size, size=2)
        
        new_points = points.copy()
        new_points[idx] += delta
        new_points[idx] = np.clip(new_points[idx], -0.5, 0.5)
        
        new_score = score(new_points)
        
        if new_score > current_score:
            points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            prob = np.exp((new_score - current_score) / T)
            if np.random.rand() < prob:
                points = new_points
                current_score = new_score
                
        T *= alpha
        step_size *= 0.999
        
    # Phase 2: Hill Climbing (Local Search)
    hc_start_time = time.time()
    hc_time_limit = 10.0
    hc_step_size = 0.01
    
    while time.time() - hc_start_time < hc_time_limit:
        idx = np.random.randint(N)
        delta = np.random.normal(0, hc_step_size, size=2)
        
        new_points = points.copy()
        new_points[idx] += delta
        new_points[idx] = np.clip(new_points[idx], -0.5, 0.5)
        
        new_score = score(new_points)
        
        if new_score > current_score:
            points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            pass
            
    return best_points

# EVOLVE-BLOCK-END
