import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = 110  # Leave 10s margin
    
    # Generate candidates with norm squared <= 4
    # This covers norms 1, sqrt(2), sqrt(3), 2
    candidates = []
    
    # Norm 1: +/- e_i
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            
    # Norm sqrt(2): +/- e_i +/- e_j
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Norm sqrt(3): +/- e_i +/- e_j +/- e_k
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)
                    
    # Norm 2: 
    # Type A: +/- 2 e_i
    for i in range(D):
        for s in (2, -2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            
    # Type B: +/- e_i +/- e_j +/- e_k +/- e_l (four 1s)
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        candidates.append(v)
                        
    candidates = np.array(candidates, dtype=np.int64)
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Target norms squared
    # We want to try R^2 = 1, 2, 3, 4
    target_norms_sq = [1, 2, 3, 4]
    
    best_set = []
    best_score = 0
    
    for R_sq in target_norms_sq:
        if time.time() > deadline:
            break
            
        # Filter candidates with norm_sq <= R_sq
        mask = norms_sq <= R_sq
        filtered_candidates = candidates[mask]
        
        if len(filtered_candidates) == 0:
            continue
            
        # Greedy packing
        # We try multiple random orders
        # To maximize speed, we do a few iterations
        
        max_iter = 50
        # If time is tight, reduce iterations
        elapsed = time.time() - start_time
        if elapsed > 80:
            max_iter = 5
        if elapsed > 100:
            max_iter = 1
            
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            indices = np.arange(len(filtered_candidates))
            np.random.shuffle(indices)
            shuffled_candidates = filtered_candidates[indices]
            
            selected = []
            selected_sq_norms = [] # Not needed, R_sq is fixed for this run
            
            for v in shuffled_candidates:
                if time.time() > deadline:
                    break
                
                # Check distance to all selected
                # dist_sq = ||v - u||^2 = ||v||^2 + ||u||^2 - 2 v.u
                # We need dist_sq >= R_sq
                
                compatible = True
                for u in selected:
                    diff = v - u
                    dist_sq = np.dot(diff, diff)
                    if dist_sq < R_sq:
                        compatible = False
                        break
                
                if compatible:
                    selected.append(v)
                    
        if len(selected) > best_score:
            best_score = len(selected)
            best_set = selected
            
    if not best_set:
        # Fallback to seed logic if something went wrong
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return np.array(best_set, dtype=np.int64)
