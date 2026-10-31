"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations, product

def construct():
    deadline = time.time() + 130  # Safety margin
    
    # All weight-2 vectors: C(11,2) * 4 = 220
    weight2_vectors = []
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    weight2_vectors.append(v)
    w2_np = np.array(weight2_vectors, dtype=np.int64)
    
    # All weight-4 vectors: C(11,4) * 16 = 330 * 16 = 5280
    w4_list = []
    for pos in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = [0] * D
            for p, s in zip(pos, signs):
                v[p] = s
            w4_list.append(v)
    w4_np = np.array(w4_list, dtype=np.int64)
    n_w4 = len(w4_np)
    
    # Conflict matrix for weight-4: conflict if inner product > 2
    ip_matrix = w4_np @ w4_np.T
    conflict = (ip_matrix > 2).astype(np.int8)
    np.fill_diagonal(conflict, 0)
    
    # Greedy independent set with multiple random orderings
    best_selected = None
    best_count = 0
    rng = np.random.RandomState(42)
    
    n_trials = 100
    for trial in range(n_trials):
        if time.time() > deadline - 15:
            break
        
        order = rng.permutation(n_w4)
        selected_mask = np.zeros(n_w4, dtype=bool)
        count = 0
        
        for idx in order:
            if time.time() > deadline - 5:
                break
            # Check conflict with all currently selected
            if not np.any(conflict[idx] & selected_mask):
                selected_mask[idx] = True
                count += 1
        
        if count > best_count:
            best_count = count
            best_selected = selected_mask.copy()
    
    # Combine weight-2 and selected weight-4
    if best_selected is not None:
        w4_selected = w4_np[best_selected]
        all_vectors = np.vstack([w2_np, w4_selected])
    else:
        all_vectors = w2_np
    
    return all_vectors
# EVOLVE-BLOCK-END
