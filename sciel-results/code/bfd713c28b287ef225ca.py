import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # 1. Generate candidate vectors
    # Supports: combinations of 3 indices
    supports = list(itertools.combinations(range(D), 3))
    
    # For each support, 4 sign patterns
    # (1,1,1), (1,1,-1), (1,-1,1), (-1,1,1)
    sign_patterns = [
        (1, 1, 1),
        (1, 1, -1),
        (1, -1, 1),
        (-1, 1, 1)
    ]
    
    candidates = []
    for sup in supports:
        for sig in sign_patterns:
            v = [0] * D
            v[sup[0]] = sig[0]
            v[sup[1]] = sig[1]
            v[sup[2]] = sig[2]
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    # Precompute dot products? n=660. 660x660 is small.
    # Dot matrix
    dots = candidates @ candidates.T
    
    # We want to select a subset S such that dots[i][j] <= 1 for all i != j in S.
    # This is an Independent Set in the graph where edges are dots[i][j] >= 2.
    
    # Greedy with randomization
    best_set = []
    best_size = 0
    
    for _ in range(50): # 50 restarts
        if time.time() > deadline:
            break
            
        # Random permutation
        perm = np.random.permutation(n)
        
        selected = []
        selected_indices = set()
        
        for idx in perm:
            # Check compatibility with all selected
            # dots[idx, selected] <= 1
            if selected:
                # Vectorized check
                sel_arr = np.array(selected)
                if np.any(dots[idx, sel_arr] >= 2):
                    continue
            selected.append(idx)
            
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected
            
        if best_size > 400: # Good enough?
             break

    # Convert best_set to array
    if best_set:
        return candidates[best_set]
    else:
        # Fallback to seed if something went wrong
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
