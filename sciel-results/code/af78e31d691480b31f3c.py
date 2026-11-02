"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def construct():
    best_vectors = []
    best_k = 0
    
    # Focus on Norm^2=4 which allows up to 330 vectors (Steiner System S(3,4,11))
    # Condition: Dot product <= 2 for Norm^2=4 vectors.
    # This is equivalent to no two vectors sharing 3 indices with same signs.
    
    candidates = []
    for i, j, k, l in combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = [0] * D
                        v[i], v[j], v[k], v[l] = si, sj, sk, sl
                        candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    # Greedy selection with dot product check
    selected = []
    max_norm_sq = 4
    
    for v in candidates:
        if not selected:
            selected.append(v)
            continue
        
        # Check dot products with selected vectors
        # v_i . v_j <= 2
        # Since all are weight 4 with +/-1, max dot product is 4.
        # Forbidden: dot product >= 3.
        
        selected_arr = np.array(selected)
        dots = np.sum(selected_arr * v, axis=1)
        
        if np.max(dots) <= 2:
            selected.append(v)
    
    if len(selected) > best_k:
        best_k = len(selected)
        best_vectors = selected

    # Fallback to D11 roots if Norm^2=4 fails (unlikely)
    if not best_vectors:
        vectors = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        best_vectors = vectors

    return np.array(best_vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
