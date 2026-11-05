import random
import numpy as np
import time

N = 64
MAX_DIST = 2 * (N - 1) ** 2

# EVOLVE-BLOCK-START
def construct():
    # Generate all candidate points
    points = [(x, y) for x in range(N) for y in range(N)]
    n_points = len(points)
    
    best_set = []
    best_size = 0
    
    # Use multiple random restarts to find a larger set
    # Relaxed constraint: only forbid isosceles triangles, not all duplicate distances
    num_restarts = 1000 
    start_time = time.time()
    time_limit = 280  # Keep buffer for 300s limit
    
    for _ in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Shuffle indices for faster permutation than shuffling list of tuples
        indices = np.random.permutation(n_points)
        S = []
        # dist_sets[i] contains squared distances from S[i] to all other points in S
        dist_sets = []
        
        for idx in indices:
            p = points[idx]
            px, py = p
            
            if not S:
                S.append(p)
                dist_sets.append(set())
                continue
            
            # Convert current set to array for vectorized distance calculation
            S_arr = np.array(S)
            p_arr = np.array([px, py])
            
            # Calculate squared distances from p to all points in S
            dists = np.sum((S_arr - p_arr) ** 2, axis=1)
            
            # 1. Check for duplicates in new distances (apex at p)
            # Convert to list for fast set check
            d_list = dists.tolist()
            if len(set(d_list)) != len(S):
                continue
            
            # 2. Check for isosceles triangles with apex at existing points
            # For each i, check if dists[i] is in dist_sets[i]
            valid = True
            for i in range(len(S)):
                if d_list[i] in dist_sets[i]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Valid point, add to S
            S.append(p)
            
            # Update dist_sets
            new_idx = len(S) - 1
            dist_sets.append(set())
            
            # Update existing sets with new distances
            for i in range(new_idx):
                d = d_list[i]
                dist_sets[i].add(d)
                dist_sets[new_idx].add(d)
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
        # Early stopping if we reach a high value
        if best_size > 200:
            break

    return best_set
# EVOLVE-BLOCK-END
