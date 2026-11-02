"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    # Generate all weight-4 vectors with ±1 entries
    # C(11,4) = 330 positions, 2^4 = 16 sign combos -> 5280 vectors
    candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_candidates = len(candidates)
    
    # Constraint: for vectors of norm 2 (norm_sq=4),
    # |u-v|^2 >= 4  <=>  4+4-2*u.v >= 4  <=>  u.v <= 2
    # So we need dot product <= 2 for all pairs
    
    # Precompute dot product matrix for all candidates
    # This is 5280 x 5280, which is ~28M entries, manageable
    dots = candidates @ candidates.T
    
    # Greedy selection with multiple random restarts
    rng = np.random.default_rng(42)
    best_set = []
    
    for restart in range(200):
        if time.time() > deadline - 10.0:
            break
        
        order = rng.permutation(n_candidates)
        selected = []
        selected_set = set()
        
        for i in order:
            if time.time() > deadline - 5.0:
                break
            
            if i in selected_set:
                continue
            
            # Check if candidate i is compatible with all selected
            # dots[i, j] <= 2 for all j in selected
            if not selected:
                selected.append(i)
                selected_set.add(i)
                continue
            
            sel_arr = np.array(selected)
            if np.all(dots[i, sel_arr] <= 2):
                selected.append(i)
                selected_set.add(i)
        
        if len(selected) > len(best_set):
            best_set = selected
            # Early termination if we hit a good target
            if len(best_set) >= 500:
                break
    
    if not best_set:
        # Fallback: return weight-2 vectors
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    result = candidates[best_set]
    
    # Final verification
    if len(result) > 1:
        from scipy.spatial.distance import cdist
        dists = cdist(result, result)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        max_norm = np.max(np.linalg.norm(result, axis=1))
        
        if min_dist < max_norm - 1e-9:
            # Fallback to weight-2 vectors
            vectors = []
            for i, j in itertools.combinations(range(D), 2):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j] = si, sj
                        vectors.append(v)
            return np.array(vectors, dtype=np.int64)
    
    return result
# EVOLVE-BLOCK-END
