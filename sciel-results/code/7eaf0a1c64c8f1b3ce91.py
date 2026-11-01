"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools
from scipy.special import comb

_start_time = time.time()
TIME_LIMIT = 115  # seconds, safety margin before 120s kill


def construct():
    global _start_time
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT:
        # Fallback: return a known valid configuration
        # Use norm-sqrt-2 vectors (2 non-zero entries)
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    # Generate all norm-2 vectors (4 non-zero entries, each +/-1)
    # Number: C(11,4) * 2^4 = 330 * 16 = 5280
    norm2_vectors = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            norm2_vectors.append(v)

    norm2_vectors = np.array(norm2_vectors, dtype=np.int64)
    n_candidates = len(norm2_vectors)  # 5280

    # Precompute squared norms (all are 4)
    # We need min pairwise distance >= max norm = 2
    # So we need ||v_i - v_j|| >= 2, i.e., ||v_i - v_j||^2 >= 4
    
    # For two vectors with 4 non-zero entries each:
    # If they share k positions with same sign: contribution 0
    # If they share k positions with opposite sign: contribution 2 (diff is +/-2)
    # If they don't share a position where both are non-zero: 
    #   v_i has 4-k non-zero where v_j is 0, and v_j has 4-k non-zero where v_i is 0
    #   Contribution from v_i-only positions: k positions where v_i=+/-1, v_j=0 -> diff^2 = 1 each
    #   Contribution from v_j-only positions: similar
    #   Plus positions where both non-zero but opposite signs: diff^2 = 4 each
    
    # Let's just do greedy selection with vectorized distance checks

    selected = []
    selected_indices = []
    
    # Shuffle for diversity? No, use fixed order for reproducibility
    order = np.arange(n_candidates)
    
    for idx in order:
        elapsed = time.time() - _start_time
        if elapsed > TIME_LIMIT:
            break
        
        if len(selected) == 0:
            selected.append(idx)
            continue
        
        # Check distance to all selected vectors
        v = norm2_vectors[idx]
        selected_vecs = norm2_vectors[selected]
        
        # Compute squared distances
        # ||v - w||^2 = ||v||^2 + ||w||^2 - 2*v.w = 4 + 4 - 2*v.w = 8 - 2*v.w
        # We need ||v - w||^2 >= 4, so 8 - 2*v.w >= 4, i.e., v.w <= 2
        
        dot_products = selected_vecs @ v
        max_dot = np.max(dot_products)
        
        if max_dot <= 2:
            selected.append(idx)
    
    if len(selected) == 0:
        # Fallback
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    result = norm2_vectors[selected]
    return result
# EVOLVE-BLOCK-END
