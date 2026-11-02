"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

DEADLINE_MARGIN = 10.0  # seconds before hard limit

def _time_left(start):
    return 120.0 - DEADLINE_MARGIN - (time.time() - start)

def construct():
    start = time.time()
    random.seed(42)
    np.random.seed(42)

    # Generate candidate pool: vectors with entries in {-1, 0, 1}, nonzero
    # Limit to manageable size by using a subset
    candidates = []
    for mask in range(1, 3**D):
        v = [0] * D
        m = mask
        for i in range(D):
            digit = m % 3
            m //= 3
            if digit == 1:
                v[i] = 1
            elif digit == 2:
                v[i] = -1
        if any(x != 0 for x in v):
            candidates.append(tuple(v))
    
    candidates = list(set(candidates))
    n_cand = len(candidates)
    
    # Precompute as numpy array for vectorized operations
    cand_arr = np.array(candidates, dtype=np.int32)
    norms_sq = np.sum(cand_arr ** 2, axis=1).astype(np.float64)
    
    best_set = []
    best_size = 0
    
    # For each possible M (max norm squared), find a large independent set
    # Only try M values that are actually present in the candidate pool
    unique_norms = sorted(set(norms_sq.tolist()))
    
    for M in unique_norms:
        if _time_left(start) < 15:
            break
        
        # Filter candidates with norm^2 <= M
        valid_mask = norms_sq <= M
        valid_indices = np.where(valid_mask)[0]
        n_valid = len(valid_indices)
        
        if n_valid < best_size + 1:
            continue
        
        # If n_valid is too large, subsample to keep computation tractable
        max_pool = 5000
        if n_valid > max_pool:
            # Randomly sample max_pool candidates
            sampled = np.random.choice(valid_indices, size=max_pool, replace=False)
            valid_indices = sampled
            n_valid = max_pool
        
        # Get the candidate vectors for this pool
        pool_arr = cand_arr[valid_indices]  # shape (n_valid, D)
        
        # Compute pairwise squared distances (n_valid x n_valid)
        # dist_sq[i,j] = ||v_i - v_j||^2
        # Use: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        norms_pool = np.sum(pool_arr ** 2, axis=1)
        dot_products = pool_arr @ pool_arr.T
        dist_sq = norms_pool[:, None] + norms_pool[None, :] - 2 * dot_products
        np.fill_diagonal(dist_sq, 0)
        
        # Conflict: dist_sq < M means we cannot have both i and j
        conflict = (dist_sq < M) & (dist_sq > 0)
        
        # Greedy independent set: repeatedly pick vertex with min degree
        degree = np.sum(conflict, axis=1).astype(np.int32)
        remaining = set(range(n_valid))
        chosen = []
        
        while remaining and _time_left(start) > 5:
            # Find min degree among remaining
            min_deg = min(degree[i] for i in remaining)
            min_candidates = [i for i in remaining if degree[i] == min_deg]
            pick = random.choice(min_candidates)
            chosen.append(pick)
            remaining.discard(pick)
            
            # Remove neighbors of pick from remaining
            neighbors = np.where(conflict[pick])[0]
            for j in neighbors:
                if j in remaining:
                    remaining.discard(j)
                    degree[j] = -1  # mark as removed
            
            # Update degrees for remaining vertices
            for j in remaining:
                if not conflict[pick, j]:
                    degree[j] -= 1
        
        # Local search: try to add more vertices
        chosen_set = set(chosen)
        for attempt in range(100):
            if _time_left(start) < 5:
                break
            improved = False
            # Try each vertex not in chosen
            for j in range(n_valid):
                if j in chosen_set:
                    continue
                # Check if j can be added (no conflicts with any chosen)
                if not np.any(conflict[j, chosen]):
                    chosen.append(j)
                    chosen_set.add(j)
                    improved = True
                    break
            if not improved:
                break
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = [valid_indices[i] for i in chosen]
    
    # Convert best_set to array
    if best_set:
        result = cand_arr[best_set].astype(np.int64)
    else:
        # Fallback: simple set of vectors with two non-zero entries
        from itertools import combinations as comb
        vectors = []
        for i, j in comb(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        result = np.array(vectors, dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
