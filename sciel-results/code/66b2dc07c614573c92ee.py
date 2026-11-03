"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

def construct():
    D = 11
    best_vectors = []
    best_k = 0
    
    # Generate candidates for norm^2 = 4
    # These are vectors with 4 non-zero entries of +/-1, or 1 entry of +/-2
    candidates = []
    
    # Type A: +/- 2 on one coordinate
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type B: +/- 1 on four coordinates
    for i, j, k, l in combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = [0] * D
                        v[i], v[j], v[k], v[l] = si, sj, sk, sl
                        candidates.append(v)
                        
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Greedy selection with random restarts to escape local optima
    num_restarts = 20
    
    for r in range(num_restarts):
        selected = []
        indices = np.arange(n_cand)
        np.random.shuffle(indices)
        
        for idx in indices:
            v = candidates[idx]
            if not selected:
                selected.append(v)
                continue
            
            selected_arr = np.array(selected)
            diffs = selected_arr - v
            dists_sq = np.sum(diffs ** 2, axis=1)
            min_dist_sq = np.min(dists_sq)
            
            if min_dist_sq >= 4:
                selected.append(v)
                
        if len(selected) > best_k:
            best_k = len(selected)
            best_vectors = selected
            
    # Fallback if no vectors found (should not happen)
    if not best_vectors:
        vectors = []
        for i in range(D):
            for s in (1, -1):
                v = [0] * D
                v[i] = s
                vectors.append(v)
        best_vectors = vectors

    return np.array(best_vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
