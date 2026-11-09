"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(points):
    """Compute the feasible radius for a set of points."""
    if points is None or len(points) == 0:
        return 0.0
    # Distance to boundary
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dist_boundary)
    
    # Pairwise distances
    from scipy.spatial.distance import pdist
    d = pdist(points)
    min_dist = np.min(d)
    min_pair = min_dist / 2.0
    
    return min(min_boundary, min_pair)

def _fast_farthest_point_sampling(n_points, seed=42):
    """
    Generate an initial set of points using Farthest Point Sampling.
    This ensures a reasonably uniform distribution, which is a good starting
    point for local optimization.
    """
    rng = np.random.default_rng(seed)
    # Start with a random point inside the unit circle
    # Using polar coordinates for uniform distribution
    r0 = np.sqrt(rng.uniform())
    theta0 = rng.uniform(0, 2 * np.pi)
    points = np.array([[r0 * np.cos(theta0), r0 * np.sin(theta0)]])
    
    for _ in range(n_points - 1):
        # Sample a large number of candidate points
        n_candidates = 5000
        r = np.sqrt(rng.uniform(size=n_candidates))
        theta = rng.uniform(0, 2 * np.pi, size=n_candidates)
        candidates = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        
        # Calculate minimum distance from each candidate to existing points
        # Using scipy for speed
        from scipy.spatial.distance import cdist
        dists = cdist(candidates, points)
        min_dists = np.min(dists, axis=1)
        
        # Select the candidate with the maximum minimum distance
        idx = np.argmax(min_dists)
        points = np.vstack([points, candidates[idx]])
        
    return points

def _optimize_packing(points, time_limit=100.0):
    """
    Deterministic gradient ascent to maximize the minimum distance.
    We maximize min( min_{i<j} |pi - pj|, min_i (1 - |pi|) ).
    Let r be the radius. We want to maximize r.
    Equivalently, minimize -r.
    
    We use a simple iterative scheme:
    1. Find the pair of points (or point-boundary) that limits the radius.
    2. Move these points slightly to increase the distance.
    """
    start_time = time.time()
    points = points.copy()
    current_r = _compute_radius(points)
    
    # Parameters
    step_size = 0.01
    decay = 0.99
    max_iter = 5000
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Compute distances
        from scipy.spatial.distance import pdist
        d = pdist(points)
        min_pair_dist = np.min(d)
        
        dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
        min_boundary_dist = np.min(dist_boundary)
        
        current_r = min(min_pair_dist / 2.0, min_boundary_dist)
        
        # Determine the constraint type
        if min_pair_dist / 2.0 < min_boundary_dist:
            # Constraint is a pair
            # Find the index of the minimum distance in the condensed distance matrix
            idx = np.argmin(d)
            # Convert condensed index to (i, j)
            # n = N, k = n*(n-1)/2
            # i = floor((sqrt(8*idx + 1) - 1) / 2)
            # j = i + 1 + (idx - i*(n-1 - i)/2)
            n = len(points)
            i = int((np.sqrt(8 * idx + 1) - 1) / 2)
            # Calculate j
            # The number of elements before row i is i*(n-1) - i*(i-1)/2
            # Actually, simpler:
            # The offset of row i is i*(n-1) - i*(i-1)/2
            # Let's just use a lookup or standard formula
            # Standard formula for condensed index k to (i,j) where i < j:
            # i = floor((sqrt(8k+1)-1)/2)
            # j = k - i*(n-1) + i*(i-1)/2 + i + 1
            j = idx - i*(n-1) + i*(i-1)//2 + i + 1
            
            p_i = points[i]
            p_j = points[j]
            
            # Vector from i to j
            v = p_j - p_i
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-10:
                # Random perturbation
                p_i += step_size * 0.1 * np.random.randn(2)
                p_j -= step_size * 0.1 * np.random.randn(2)
            else:
                # Unit vector
                u = v / norm_v
                
                # Move i away from j, j away from i
                # But we must also respect boundaries
                # Simple move
                move_i = -u * step_size
                move_j = u * step_size
                
                # Check if moving i out of bounds
                new_p_i = p_i + move_i
                if np.linalg.norm(new_p_i) > 1.0 - current_r:
                    # Project back or reduce step
                    # For simplicity, just don't move if it violates boundary significantly
                    # Or move radially outward? No, we want to increase separation.
                    # If i is close to boundary, maybe move j more?
                    # Let's just scale down the move if it goes out
                    dist_to_bound = 1.0 - np.linalg.norm(p_i)
                    if dist_to_bound < step_size:
                        move_i *= dist_to_bound / step_size * 0.5
                        move_j *= 1.0 # Compensate? Or just let it be
                    
                new_p_j = p_j + move_j
                if np.linalg.norm(new_p_j) > 1.0 - current_r:
                    dist_to_bound = 1.0 - np.linalg.norm(p_j)
                    if dist_to_bound < step_size:
                        move_j *= dist_to_bound / step_size * 0.5
                        move_i *= 1.0

                points[i] += move_i
                points[j] += move_j
                
        else:
            # Constraint is boundary
            idx = np.argmin(dist_boundary)
            p_idx = points[idx]
            
            # Move point towards center
            norm_p = np.linalg.norm(p_idx)
            if norm_p > 1e-10:
                u = p_idx / norm_p
                move = -u * step_size
                points[idx] += move
            else:
                points[idx] += step_size * 0.1 * np.random.randn(2)
        
        # Decay step size slowly
        if it % 100 == 0:
            step_size *= decay

    return points

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin
    
    # 1. Generate initial points using FPS
    # FPS is fast and gives a good spread
    points = _fast_farthest_point_sampling(N)
    
    # 2. Optimize
    optimized_points = _optimize_packing(points, time_limit=100.0)
    
    # 3. Final check and return
    r_opt = _compute_radius(optimized_points)
    r_init = _compute_radius(points)
    
    if r_opt > r_init:
        return optimized_points
    else:
        return points

# EVOLVE-BLOCK-END
