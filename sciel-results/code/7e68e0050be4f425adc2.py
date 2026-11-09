import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    best_vectors = None
    best_size = 0
    
    # Strategy: Generate a large pool of sparse integer vectors.
    # Include vectors with 1, 2, 3, 4 non-zeros.
    # For 1 non-zero: 22 vectors (norm 1)
    # For 2 non-zeros: 2*55*4 = 440 vectors (norm sqrt(2))
    # For 3 non-zeros: 2*165*8 = 2640 vectors (norm sqrt(3))
    # For 4 non-zeros: 2*330*16 = 10560 vectors (norm sqrt(4)=2)
    # Total ~ 13662 vectors.
    
    D = 11
    candidates = []
    
    # Generate 1-nonzero vectors
    for i in range(D):
        candidates.append(np.array([0]*i + [1] + [0]*(D-i-1)))
        candidates.append(np.array([0]*i + [-1] + [0]*(D-i-1)))
        
    # Generate 2-nonzero vectors
    for i, j in itertools.combinations(range(D), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Generate 3-nonzero vectors
    for i, j, k in itertools.combinations(range(D), 3):
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)
                    
    # Generate 4-nonzero vectors
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    for sl in [1, -1]:
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        candidates.append(v)
                        
    candidates = np.array(candidates)
    N = len(candidates)
    
    # Greedy selection with random restarts
    # We will try a few different random orderings and keep the best result.
    # To speed up, we can pre-compute norms.
    norms = np.linalg.norm(candidates, axis=1)
    
    # The condition is: min pairwise distance >= max norm of the set.
    # Let R be the max norm in the selected set.
    # Then for all i, j in set, ||v_i - v_j|| >= R.
    # Note: ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2 v_i . v_j.
    
    # Since we mix vectors of different norms, R will be the max norm among selected.
    # If we select any 4-nonzero vector, R >= 2.
    # If we only select 3-nonzero, R >= sqrt(3) ~ 1.732.
    # If we only select 2-nonzero, R >= sqrt(2) ~ 1.414.
    
    # Let's try to maximize k.
    # We can run greedy multiple times with different seeds.
    
    best_set = []
    
    for seed in range(10):
        if time.time() > deadline:
            break
            
        np.random.seed(seed)
        perm = np.random.permutation(N)
        
        selected_indices = []
        selected_vectors = []
        current_max_norm = 0
        
        for idx in perm:
            if time.time() > deadline:
                break
                
            v = candidates[idx]
            v_norm = norms[idx]
            
            if not selected_vectors:
                selected_indices.append(idx)
                selected_vectors.append(v)
                current_max_norm = v_norm
                continue
            
            # Check if adding v violates constraints
            # 1. New max norm R' = max(current_max_norm, v_norm)
            # 2. Distances to all existing vectors must be >= R'
            
            new_max_norm = max(current_max_norm, v_norm)
            
            # Vectorized distance check
            # dists = np.linalg.norm(np.array(selected_vectors) - v, axis=1)
            # This is O(K) per candidate.
            # If K is large, this is slow.
            # But N is ~13k. If K is ~100, this is 1.3M ops. Fast.
            
            sel_arr = np.array(selected_vectors)
            dists = np.linalg.norm(sel_arr - v, axis=1)
            
            if np.min(dists) >= new_max_norm - 1e-9:
                selected_indices.append(idx)
                selected_vectors.append(v)
                current_max_norm = new_max_norm
                
        if len(selected_vectors) > len(best_set):
            best_set = selected_vectors.copy()
            
    if best_set:
        return np.array(best_set)
    else:
        # Fallback: return a single vector if nothing found (should not happen)
        return np.array(candidates[0]).reshape(1, -1)
