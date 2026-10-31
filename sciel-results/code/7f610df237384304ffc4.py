import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # 10 seconds safety margin
    
    D = 11
    candidates = []
    
    # Generate all integer vectors with norm squared <= 4
    # Possible non-zero entries:
    # 1. One entry is +/- 1 (norm sq 1)
    # 2. One entry is +/- 2 (norm sq 4)
    # 3. Two entries are +/- 1 (norm sq 2)
    # 4. Three entries are +/- 1 (norm sq 3)
    # 5. Four entries are +/- 1 (norm sq 4)
    
    # Type 1: Weight 1, value +/- 1
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            
    # Type 2: Weight 1, value +/- 2
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
            
    # Type 3: Weight 2, values +/- 1
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Type 4: Weight 3, values +/- 1
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    # Type 5: Weight 4, values +/- 1
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    norms_sq = np.sum(candidates**2, axis=1)
    # Max norm squared is 4
    max_norm_sq = 4
    
    dots = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflict = (dist_sq < max_norm_sq)
    np.fill_diagonal(conflict, False)
    
    best_selected = []
    best_size = 0
    
    np.random.seed(42)
    
    # Run multiple restarts with time guard
    attempt = 0
    while time.time() < deadline:
        remaining = np.ones(n, dtype=bool)
        degrees = conflict.sum(axis=1).copy()
        selected = []
        
        while remaining.any():
            if time.time() > deadline - 5:  # Check deadline inside loop
                break
                
            rem_indices = np.where(remaining)[0]
            
            if len(rem_indices) == 0:
                break
            
            rem_degrees = degrees[rem_indices]
            min_deg = np.min(rem_degrees)
            
            min_mask = rem_degrees == min_deg
            min_indices_local = np.where(min_mask)[0]
            
            chosen_local = np.random.choice(min_indices_local)
            chosen_global = rem_indices[chosen_local]
            
            selected.append(chosen_global)
            
            neigh_mask = conflict[chosen_global]
            
            to_remove_mask = np.zeros(n, dtype=bool)
            to_remove_mask[chosen_global] = True
            to_remove_mask[neigh_mask] = True
            
            actual_remove = to_remove_mask & remaining
            
            if np.any(actual_remove):
                deg_decrease = conflict[actual_remove][:, remaining].sum(axis=0)
                degrees[remaining] -= deg_decrease
                
            remaining[actual_remove] = False
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected
        
        attempt += 1
    
    # If no valid selection was found (shouldn't happen), return a single vector
    if len(best_selected) == 0:
        best_selected = [0]
    
    result = candidates[best_selected]
    return result
