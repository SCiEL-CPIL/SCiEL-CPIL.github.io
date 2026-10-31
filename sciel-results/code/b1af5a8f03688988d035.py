import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    D = 11
    candidates = []
    
    # Type 1: two ±1's, rest 0 (norm^2 = 2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Type 2: one ±2, rest 0 (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
    
    # Type 3: four ±1's, rest 0 (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    norms_sq = np.sum(candidates**2, axis=1)
    
    # Build conflict matrix
    dots = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflict = (dist_sq < 4)
    np.fill_diagonal(conflict, False)
    
    # Pure Type 1 baseline
    mask_t1 = norms_sq == 2
    t1_indices = np.where(mask_t1)[0]
    best_size = len(t1_indices)
    best_selected = list(t1_indices)
    
    # Fast greedy using numpy
    np.random.seed(42)
    
    while time.time() < deadline - 2:
        remaining = np.ones(n, dtype=bool)
        # Compute degrees for remaining nodes
        deg = conflict.sum(axis=1).astype(np.int32)
        selected = []
        
        while remaining.any():
            rem_idx = np.where(remaining)[0]
            if len(rem_idx) == 0:
                break
            
            rem_deg = deg[rem_idx]
            min_d = rem_deg.min()
            min_mask = rem_deg == min_d
            min_local = np.where(min_mask)[0]
            
            chosen_local = np.random.choice(min_local)
            chosen = rem_idx[chosen_local]
            selected.append(chosen)
            
            # Remove chosen and neighbors
            remove_mask = conflict[chosen].copy()
            remove_mask[chosen] = True
            to_remove = remove_mask & remaining
            
            if to_remove.any():
                # Update degrees
                deg[remaining] -= conflict[to_remove][:, remaining].sum(axis=0)
            
            remaining[to_remove] = False
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected
    
    result = candidates[best_selected]
    return result
