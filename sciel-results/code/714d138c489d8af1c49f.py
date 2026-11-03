<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np


def construct():
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Start with the seed configuration: all pairs (i,j) with entries ±1
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
    
    V = np.array(vectors, dtype=np.int64)
    
    # Compute max norm squared
    norms_sq = np.sum(V ** 2, axis=1)
    max_norm_sq = int(np.max(norms_sq))
    
    # Generate candidate vectors with entries in {-1, 0, 1}
    candidates = []
    for combo in product([-1, 0, 1], repeat=D):
        v = np.array(combo, dtype=np.int64)
        if np.sum(v ** 2) == 0:
            continue
        candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    
    # Filter out candidates already in V
    # Create a set of tuples for fast lookup
    V_set = set(map(tuple, V))
    new_candidates = []
    for v in candidates:
        if tuple(v) not in V_set:
            new_candidates.append(v)
    
    new_candidates = np.array(new_candidates, dtype=np.int64)
    
    # Try to add valid candidates
    V_list = list(V)
    max_norm_sq_current = max_norm_sq
    
    for v in new_candidates:
        if time.time() > deadline:
            break
        
        # Check if v can be added: min ||v - u||^2 >= max_norm_sq_current
        V_arr = np.array(V_list, dtype=np.int64)
        diffs = V_arr - v
        dists_sq = np.sum(diffs ** 2, axis=1)
        min_dist_sq = int(np.min(dists_sq))
        
        if min_dist_sq >= max_norm_sq_current:
            V_list.append(v)
            # Update max norm squared if needed
            v_norm_sq = int(np.sum(v ** 2))
            if v_norm_sq > max_norm_sq_current:
                max_norm_sq_current = v_norm_sq
    
    V_final = np.array(V_list, dtype=np.int64)
    return V_final
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
