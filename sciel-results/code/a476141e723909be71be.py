import numpy as np
import itertools
import time
import random

D = 11

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 120.0  # 120s limit, keep margin
    
    # Generate all weight 4 vectors
    # Indices 0..10
    # Signs +/-
    candidates = []
    for coords in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for c, s in zip(coords, signs):
                v[c] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    N = len(candidates)
    
    # We want max subset such that dot(u, v) <= 2 for all u != v
    # Note: dot(u, u) = 4. We exclude self.
    
    best_set = []
    
    # Greedy with random restarts
    # To speed up, we can use a heuristic ordering or just random shuffle
    
    max_restarts = 100
    
    for r in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        # Shuffle candidates
        idx = np.arange(N)
        np.random.shuffle(idx)
        
        selected = []
        selected_indices = []
        
        # For efficiency, we can maintain a set of "blocked" indices
        # But N=5280, N^2 is ~28M ops. Python loop is slow.
        # Vectorized dot product check is better.
        
        # Let's try a simple greedy loop
        # To make it faster, we can pre-compute dot products? 
        # 5280x5280 matrix is 28M floats. 224MB. Might be too big/slow to allocate repeatedly.
        # But we can do it once.
        
        # Actually, let's just do a few restarts with a simple greedy.
        
        for i in idx:
            v = candidates[i]
            # Check against all selected
            # If selected is small, this is fast.
            # If selected is large, this is slow.
            
            # Optimization: Only check if potential to improve?
            # No, greedy just adds if valid.
            
            valid = True
            for j in selected:
                if np.dot(v, candidates[j]) > 2:
                    valid = False
                    break
            if valid:
                selected.append(i)
                
        if len(selected) > len(best_set):
            best_set = selected
            
    # Add weight 1 vectors
    # All weight 1 vectors are compatible with weight 4 (dot <= 1 <= 2? 
    # Wait, condition for V1 vs V4 is u.v <= 1.
    # u is e_i. v has v_i = +/- 1. dot is +/- 1. 
    # Max dot is 1. Condition u.v <= 1 is satisfied.
    # So we can add all 22 weight 1 vectors.
    
    final_vectors = [candidates[i] for i in best_set]
    
    # Generate weight 1 vectors
    for i in range(D):
        v = [0] * D
        v[i] = 1
        final_vectors.append(v)
        v[i] = -1
        final_vectors.append(v)
        
    return np.array(final_vectors, dtype=np.int64)
