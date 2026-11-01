"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random

def _compute_min_dist(centers):
    """
    Compute the minimum distance between any two sphere centers
    and the minimum distance from any center to the cube boundary.
    Returns the overall minimum distance (which is 2*r).
    """
    n = centers.shape[0]
    # Pairwise distances
    # Compute squared distances to avoid sqrt until necessary, but sqrt is fine for N=50
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to boundaries
    # Centers are in [-0.5, 0.5]
    # Distance to boundary for a point c is min(0.5 - c, c + 0.5) for each dimension
    # Since c is in [-0.5, 0.5], 0.5 - abs(c) is the distance to the nearest wall in that dimension.
    # The distance to the boundary of the cube is the min over all dimensions.
    dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary_dist = np.min(dist_to_boundary)
    
    return min(min_pair_dist, min_boundary_dist)

def _local_search(centers, max_iter=2000):
    """
    Local search that tries to increase the minimum distance.
    Strategy: Find the 'worst' constraint (min distance pair or boundary),
    and nudge one of the involved points to alleviate it.
    """
    centers = centers.copy()
    current_min_dist = _compute_min_dist(centers)
    
    # We will try to move points to increase the min distance.
    # A simple approach: repeatedly find the pair with min distance,
    # and move them apart. Or find the point closest to boundary and push it in.
    
    for _ in range(max_iter):
        n = centers.shape[0]
        
        # Calculate all pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Find min pair
        min_pair_val = np.min(dists)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_pair_idx
        
        # Calculate boundary distances
        dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
        min_bound_val = np.min(dist_to_boundary)
        min_bound_idx = np.argmin(dist_to_boundary)
        
        # Determine which constraint is active (limiting)
        if min_pair_val < min_bound_val:
            # Move i and j apart
            # Direction from i to j
            vec = centers[j] - centers[i]
            norm_vec = np.linalg.norm(vec)
            if norm_vec < 1e-12:
                # Random direction
                dir_vec = np.random.randn(3)
                dir_vec /= np.linalg.norm(dir_vec)
            else:
                dir_vec = vec / norm_vec
            
            # Try moving i in -dir_vec and j in +dir_vec
            # Step size proportional to current step or small fixed
            step = 0.01 
            new_i = centers[i] - step * dir_vec
            new_j = centers[j] + step * dir_vec
            
            # Check bounds
            if np.all(np.abs(new_i) < 0.5) and np.all(np.abs(new_j) < 0.5):
                centers_new = centers.copy()
                centers_new[i] = new_i
                centers_new[j] = new_j
                new_min_dist = _compute_min_dist(centers_new)
                
                if new_min_dist > current_min_dist:
                    centers = centers_new
                    current_min_dist = new_min_dist
                else:
                    # Try smaller step or random nudge if stuck?
                    # For simplicity, just try a random small nudge on i or j
                    # But to keep it deterministic-ish and fast, we might just break if no improvement found in many steps?
                    # Let's try a random perturbation on the worst point if the direct push fails.
                    pass
        else:
            # Move point min_bound_idx away from boundary
            # Determine which face is closest
            c = centers[min_bound_idx]
            # dist_to_boundary is min(0.5 - abs(c[k]))
            # Find the dimension and sign
            min_dist_dim = np.inf
            min_dim = 0
            min_sign = 0
            for k in range(3):
                d_pos = 0.5 - c[k]
                d_neg = 0.5 + c[k]
                if d_pos < min_dist_dim:
                    min_dist_dim = d_pos
                    min_dim = k
                    min_sign = 1 # Push towards -c[k] direction? No, if c[k] is positive, boundary is at 0.5, so push towards 0.
                    # Wait, distance is 0.5 - abs(c[k]).
                    # If c[k] > 0, dist is 0.5 - c[k]. To increase dist, decrease c[k]. Direction is -1.
                    # If c[k] < 0, dist is 0.5 + c[k]. To increase dist, increase c[k]. Direction is +1.
                if d_neg < min_dist_dim:
                    min_dist_dim = d_neg
                    min_dim = k
                    min_sign = -1 # If c[k] < 0, push towards 0, so +1? 
                    # Let's re-evaluate.
                    # If c[k] is close to 0.5, we want to move it towards 0. Direction -1.
                    # If c[k] is close to -0.5, we want to move it towards 0. Direction +1.
                    # So direction is -sign(c[k]).
            
            # Correct logic:
            # The closest boundary is at +0.5 if c[k] > 0, or -0.5 if c[k] < 0.
            # We want to move c[k] towards 0.
            # So direction is -sign(c[k]).
            dir_vec = np.zeros(3)
            # Find the dimension with min distance
            dists_dims = 0.5 - np.abs(c)
            k_min = np.argmin(dists_dims)
            dir_vec[k_min] = -np.sign(c[k_min])
            if dir_vec[k_min] == 0:
                dir_vec[k_min] = 1 # If exactly 0, push randomly? Unlikely.
            
            step = 0.01
            new_c = centers[min_bound_idx] + step * dir_vec
            
            if np.all(np.abs(new_c) < 0.5):
                centers_new = centers.copy()
                centers_new[min_bound_idx] = new_c
                new_min_dist = _compute_min_dist(centers_new)
                
                if new_min_dist > current_min_dist:
                    centers = centers_new
                    current_min_dist = new_min_dist
        
        # Optional: Add a random perturbation to escape local minima
        if _ % 50 == 0:
             i = np.random.randint(n)
             delta = np.random.randn(3) * 0.005
             new_c = centers[i] + delta
             if np.all(np.abs(new_c) < 0.5):
                 centers_new = centers.copy()
                 centers_new[i] = new_c
                 new_min_dist = _compute_min_dist(centers_new)
                 if new_min_dist > current_min_dist:
                     centers = centers_new
                     current_min_dist = new_min_dist

    return centers

def construct():
    best_centers = None
    best_radius = 0.0
    
    # Multi-start random initialization
    n_starts = 100
    np.random.seed(42)
    
    for _ in range(n_starts):
        # Random uniform initialization
        centers = np.random.uniform(-0.5, 0.5, (N, 3))
        
        # Local search
        optimized_centers = _local_search(centers, max_iter=1000)
        
        # Calculate radius
        min_dist = _compute_min_dist(optimized_centers)
        radius = min_dist / 2.0
        
        if radius > best_radius:
            best_radius = radius
            best_centers = optimized_centers
            
        # Early termination if we reach a very high value (optional)
        # Known best is ~0.13595
        if best_radius > 0.135:
            # We can continue to try to beat it, but this is a good sign
            pass
            
    return best_centers

# EVOLVE-BLOCK-END
