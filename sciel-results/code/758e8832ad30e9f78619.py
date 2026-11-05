import time
import random
import math
import numpy as np

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 295  # Safety margin for 330s kill
    
    # Precompute squared distances between all pairs of points in [0, N)^2
    # Points are indexed by p = x * N + y
    all_points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    
    # dist_matrix[i, j] = squared distance between point i and point j
    # Shape: (N*N, N*N)
    # Memory: 4096 * 4096 * 4 bytes = 64 MB. Acceptable.
    dx = all_points[:, 0].reshape(-1, 1) - all_points[:, 0].reshape(1, -1)
    dy = all_points[:, 1].reshape(-1, 1) - all_points[:, 1].reshape(1, -1)
    dist_matrix = dx * dx + dy * dy
    
    best_set = []
    best_size = 0
    
    # Helper to check if a candidate point can be added to a current set
    # current_set: list of point indices
    # dist_sets: list of sets of squared distances from each point in current_set to others
    # Returns: (bool, list_of_new_dist_sets) if valid, else (False, None)
    def can_add(candidate_idx, current_set, dist_sets):
        if not current_set:
            # First point, always valid
            return True, [set()]
        
        # 1. Distances from candidate to all existing points must be distinct
        # Get distances from candidate to all points in current_set
        # candidate_idx is an index into dist_matrix
        dists_from_cand = dist_matrix[candidate_idx, current_set]
        
        # Check for duplicates in dists_from_cand
        if len(set(dists_from_cand)) != len(dists_from_cand):
            return False, None
        
        # 2. For each existing point q, distance from q to candidate must not be in dist_sets[q]
        # dists_from_cand[i] is distance from current_set[i] to candidate
        for i, q_idx in enumerate(current_set):
            d_sq = dists_from_cand[i]
            if d_sq in dist_sets[i]:
                return False, None
        
        # If we reach here, it's valid. Construct the new dist_sets
        # This is the expensive part, so we only do it if valid.
        new_dist_sets = [set(s) for s in dist_sets]
        
        # Update existing sets
        for i, q_idx in enumerate(current_set):
            new_dist_sets[i].add(dists_from_cand[i])
        
        # Create set for candidate
        cand_set = set(dists_from_cand)
        new_dist_sets.append(cand_set)
        
        return True, new_dist_sets

    # We will try several random restarts
    # Strategy: Greedy construction, then local search (swaps)
    for attempt in range(2000):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Shuffle indices
        indices = np.arange(N * N)
        np.random.shuffle(indices)
        
        current_set = []
        dist_sets = []
        
        # Greedy phase
        # We can optimize this by just iterating through shuffled indices
        for idx in indices:
            if time.time() > deadline:
                break
            valid, new_ds = can_add(int(idx), current_set, dist_sets)
            if valid:
                current_set.append(int(idx))
                dist_sets = new_ds
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
        # Local Search Phase
        # Try to improve by swapping points
        # We need to track which points are in the set for fast removal?
        # Actually, removal is hard with the current dist_sets structure.
        # A swap: remove i, add j.
        # Removing i requires rebuilding dist_sets or updating them carefully.
        
        # Alternative: Simulated Annealing or Hill Climbing on the set of points.
        # Given the complexity of maintaining dist_sets during removal, 
        # let's stick to a simpler local search: 
        # Try adding a point. If it fails, try removing one existing point and adding a new one.
        
        # To make this efficient, we can precompute for each point in current_set
        # which "constraints" it imposes.
        
        # Let's try a few iterations of "remove one, try to add two" or "swap"
        # But rebuilding dist_sets is O(k^2) where k is set size.
        
        # Let's try a simple hill climbing:
        # Randomly pick a point to remove.
        # Rebuild dist_sets for the remaining points (O(k^2)).
        # Then try to add points greedily again? That's basically a restart.
        
        # Better: 
        # We can try to add points that were rejected during greedy phase.
        
        # For now, the greedy phase with many restarts is the main driver.
        # The "local search" idea from inspiration was key. 
        # Let's implement a simple swap-based local search.
        
        # To support removal efficiently, we can store the set as a boolean array 
        # and recompute dist_sets from scratch when we remove a point? 
        # That's O(k^2) per removal. If k=50, that's 2500 ops. Very fast.
        
        # Let's try:
        # 1. Greedy fill.
        # 2. Local search loop:
        #    - Pick random i in current_set to remove.
        #    - Remove i. (Recompute dist_sets for remaining k-1 points)
        #    - Try to add random points from the pool until stuck or improved.
        #    - If improved, keep. Else, revert.
        
        # This might be slow if we recompute too often.
        
        # Let's try a different local search:
        # Keep the set. Try to add a point p.
        # If it fails because of a conflict with q, try removing q and adding p.
        
        # This is getting complex. Let's rely on the speed of the greedy phase 
        # and the high number of restarts enabled by the vectorized distance check.
        
        # The vectorized check `dists_from_cand = dist_matrix[candidate_idx, current_set]` 
        # is very fast. The bottleneck is `set(dists_from_cand)` and the loop 
        # `for i, q_idx in enumerate(current_set)`.
        
        # We can optimize the loop by using numpy for the second check too?
        # `dists_from_cand` is an array.
        # `dist_sets` is a list of sets.
        # We can't easily vectorize set membership.
        
        # However, we can skip the `set(dists_from_cand)` check if we know 
        # the distances are likely distinct? No, we must check.
        
        # Let's just run many restarts.
        # If best_size is high, we might break early? 
        # No, let's use all time.

    # Convert indices back to (x, y) coordinates
    final_points = [tuple(all_points[i]) for i in best_set]
    return final_points
# EVOLVE-BLOCK-END
