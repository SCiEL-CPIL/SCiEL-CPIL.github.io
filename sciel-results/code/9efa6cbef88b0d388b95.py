"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Generate candidate vectors from structured families
    candidates = []
    
    # Type 1: two nonzero coords ±1,±1 (norm_sq=2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates.append(v)
    
    # Type 2: one nonzero coord ±2 (norm_sq=4)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
    
    # Type 3: three nonzero coords ±1,±1,±1 (norm_sq=3)
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    candidates.append(v)
    
    # Type 4: two nonzero coords ±1,±2 (norm_sq=5)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, 2 * sj
                candidates.append(v)
                candidates.append([-x for x in v])  # include negatives
    
    candidates = np.array(candidates, dtype=np.int64)
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    best_vectors = []
    best_count = 0
    
    # Try different max_norm_sq thresholds
    for max_norm_sq in [2, 3, 4, 5, 6, 8]:
        if time.time() > deadline - 30.0:
            break
        
        mask = (norms_sq <= max_norm_sq) & (norms_sq >= 2)
        filtered = candidates[mask]
        if len(filtered) == 0:
            continue
        
        # Greedy selection with vectorized distance check
        selected = []
        
        for idx in range(len(filtered)):
            if time.time() > deadline - 15.0:
                break
            
            v = filtered[idx]
            
            if len(selected) == 0:
                selected.append(v)
                continue
            
            # Vectorized check: compute distances to all selected
            sel_arr = np.array(selected, dtype=np.int64)
            diffs = sel_arr - v[np.newaxis, :]
            dist_sqs = np.sum(diffs * diffs, axis=1)
            
            if np.all(dist_sqs >= max_norm_sq):
                selected.append(v)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_vectors = selected[:]
    
    if best_vectors:
        return np.array(best_vectors, dtype=np.int64)
    
    # Fallback
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
