import numpy as np
import random
import time

N = 64
NUM_POINTS = N * N
DEADLINE = time.time() + 290  # Safety margin before 300s limit

# Pre-compute coordinates
points = np.array([(x, y) for y in range(N) for x in range(N)], dtype=np.int32)

# Pre-compute full distance matrix (4096x4096)
# This takes ~1-2 seconds and ~64MB memory
def _precompute_distances():
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sum(diff**2, axis=2).astype(np.int32)

D_grid = _precompute_distances()

def is_valid_set(S_indices):
    """
    Check if the set of indices S_indices forms a no-isosceles set.
    Returns True if valid.
    """
    if len(S_indices) <= 2:
        return True
    
    K = len(S_indices)
    # Extract submatrix
    D = D_grid[np.ix_(S_indices, S_indices)]
    
    # Check for isosceles triangles
    # Triplet (i, j, k) is isosceles if D[i,j]==D[i,k] or D[i,j]==D[j,k] or D[i,k]==D[j,k]
    for i in range(K):
        for j in range(i + 1, K):
            d_ij = D[i, j]
            for k in range(j + 1, K):
                d_ik = D[i, k]
                d_jk = D[j, k]
                if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                    return False
    return True

def construct():
    start_time = time.time()
    best_set = []
    
    # Pre-shuffle indices for randomization
    all_indices = list(range(NUM_POINTS))
    
    # We will run multiple restarts of a greedy + local search algorithm
    while time.time() < DEADLINE - 10:  # Safety margin
        
        random.shuffle(all_indices)
        S = []
        S_set = set()
        # dist_sets[s_idx] is a set of squared distances from s_idx to all other points in S
        dist_sets = {}
        
        # Helper functions for fast add/remove/check
        def is_compatible_fast(p_idx):
            if not S:
                return True
            
            # Get distances from p_idx to all points in S
            d_to_S = D_grid[p_idx, S]
            
            # Check 1: d(p, S_i) == d(p, S_j)
            # If any duplicate in d_to_S, return False
            if len(np.unique(d_to_S)) < len(S):
                return False
            
            # Check 2 & 3: d(p, S_i) == d(S_i, S_j)
            # For each S_i, check if d(p, S_i) is in dist_sets[S_i]
            for i, s_idx in enumerate(S):
                d_ps = d_to_S[i]
                if d_ps in dist_sets[s_idx]:
                    return False
            
            return True

        def add_point(p_idx):
            S.append(p_idx)
            S_set.add(p_idx)
            
            new_dist_set = set()
            for s_idx in S[:-1]: # All points except the one just added
                d = D_grid[p_idx, s_idx]
                dist_sets[s_idx].add(d)
                new_dist_set.add(d)
            dist_sets[p_idx] = new_dist_set

        def remove_point(p_idx):
            S.remove(p_idx)
            S_set.remove(p_idx)
            
            p_dist_set = dist_sets.pop(p_idx)
            for s_idx in S:
                d = D_grid[p_idx, s_idx]
                dist_sets[s_idx].discard(d)

        # Greedy Phase
        for idx in all_indices:
            if is_compatible_fast(idx):
                add_point(idx)
        
        # Local Search Phase
        pool = [i for i in all_indices if i not in S_set]
        
        # Run local search for a fixed number of steps or time
        local_search_deadline = time.time() + 5
        while time.time() < local_search_deadline and len(pool) > 0:
            # Try to add a random point from pool
            p_idx = random.choice(pool)
            
            if is_compatible_fast(p_idx):
                add_point(p_idx)
                pool.remove(p_idx)
            else:
                # Try to swap: remove one random point from S, then try to add p_idx
                # If we can add p_idx after removal, size is same.
                # To increase size, we would need to add 2 points for 1 removal, 
                # or find a point that was blocked by the removed point.
                
                # Simple heuristic: Random restart of local search if stuck?
                # Or just try to remove a point and see if we can add 2 points.
                if S:
                    rem_idx = random.choice(S)
                    remove_point(rem_idx)
                    pool.append(rem_idx)
                    
                    # Try to add 2 points from pool
                    added_count = 0
                    # We need to be careful: adding one might block the other.
                    # Try random order.
                    attempts = list(pool)
                    random.shuffle(attempts)
                    
                    for cand_idx in attempts:
                        if added_count >= 2:
                            break
                        if cand_idx in S_set:
                            continue
                        if is_compatible_fast(cand_idx):
                            add_point(cand_idx)
                            if cand_idx in pool:
                                pool.remove(cand_idx)
                            added_count += 1
                            
                    # If we added fewer than 1, we lost size. 
                    # If we added 1, size is same.
                    # If we added 2, size increased by 1.
        
        if len(S) > len(best_set):
            best_set = S[:]
            # Verification is expensive, skip during search, do at end
            # if not is_valid_set(best_set):
            #     best_set = []
            
        # If we found a very large set, we might break early?
        if len(best_set) > 90:
            break

    # Final verification
    if best_set and not is_valid_set(best_set):
         # Fallback to a small valid set if something went wrong
         return [(0, 0)]
         
    return [tuple(points[i]) for i in best_set]
