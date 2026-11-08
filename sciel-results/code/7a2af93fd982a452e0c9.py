"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

DEADLINE = 110  # seconds, safety margin before 120s limit


def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    # Generate candidate vectors of various weights
    candidates = []
    for w in [2, 3, 4]:
        for support in combinations(range(D), w):
            for signs in product([1, -1], repeat=w):
                v = np.zeros(D, dtype=np.int64)
                for i, s in zip(support, signs):
                    v[i] = s
                candidates.append(v)
    
    candidates = np.array(candidates)
    n = len(candidates)
    
    # Compute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Greedy selection: start with weight-2 (mutually compatible), then add others
    selected_indices = []
    selected_vecs = []
    
    # Add all weight-2 vectors first
    for i in range(n):
        if norms_sq[i] == 2:
            selected_indices.append(i)
            selected_vecs.append(candidates[i])
    
    # Sort remaining candidates by norm, try to add them
    remaining = [i for i in range(n) if i not in set(selected_indices)]
    remaining.sort(key=lambda i: norms_sq[i])
    
    sel_arr = np.array(selected_vecs) if selected_vecs else np.array([]).reshape(0, D)
    sel_norms = np.array([norms_sq[i] for i in selected_indices])
    
    for idx in remaining:
        if elapsed() > DEADLINE:
            break
        
        v = candidates[idx]
        vn = norms_sq[idx]
        
        if len(sel_arr) == 0:
            selected_indices.append(idx)
            selected_vecs.append(v)
            sel_arr = np.array(selected_vecs)
            sel_norms = np.array([vn])
            continue
        
        # Check: |v - w|² >= max(vn, wn) for all w in selected
        diffs = sel_arr - v
        dists_sq = np.sum(diffs * diffs, axis=1)
        max_norms = np.maximum(vn, sel_norms)
        
        if np.all(dists_sq >= max_norms):
            selected_indices.append(idx)
            selected_vecs.append(v)
            sel_arr = np.vstack([sel_arr, v.reshape(1, -1)])
            sel_norms = np.append(sel_norms, vn)
    
    if not selected_vecs:
        return np.array([[1] + [0] * (D - 1)], dtype=np.int64)
    
    return np.array(selected_vecs, dtype=np.int64)
# EVOLVE-BLOCK-END
