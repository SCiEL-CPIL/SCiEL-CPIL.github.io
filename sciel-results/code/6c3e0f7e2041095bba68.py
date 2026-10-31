import itertools
import numpy as np
import random
import time

D = 11

def construct():
    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    # Generate all weight-3 vectors
    # Indices 0..10
    # Signs 1, -1
    candidates = []
    for idxs in itertools.combinations(range(D), 3):
        i, j, k = idxs
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    v = [0] * D
                    v[i] = s1
                    v[j] = s2
                    v[k] = s3
                    candidates.append(np.array(v, dtype=np.int8))
    
    n = len(candidates)
    # n = 1320
    
    # Precompute dot products?
    # Matrix 1320x1320 of int8/16. 1.7M entries. 3.4MB. Fine.
    # Or just compute on the fly in the greedy loop?
    # Greedy loop:
    #   Pick v.
    #   Check all u in available. If dot > 1, remove u.
    #   This is O(N^2) total if we are careful.
    
    best_set = []
    
    # Try a few random restarts
    # Or just one greedy pass might be enough if we order well?
    # Let's try multiple greedy passes with random shuffling.
    
    num_restarts = 0
    while time.time() - start_time < deadline:
        num_restarts += 1
        indices = list(range(n))
        random.shuffle(indices)
        
        selected_idx = []
        available = set(indices)
        
        for i in indices:
            if time.time() - start_time > deadline:
                break
            if i not in available:
                continue
            
            # Select i
            selected_idx.append(i)
            v = candidates[i]
            
            # Remove i and all conflicts
            to_remove = [i]
            for j in available:
                if j == i:
                    continue
                # Check conflict
                # dot(v, candidates[j])
                # Using numpy dot is slow in loop.
                # Manual dot product for sparse vectors?
                # Or just use numpy if we precomputed?
                pass
            
            # This loop is slow in Python if done naively.
            # Let's precompute the conflict matrix.
            
            break # Break inner loop to go to precomputation logic? No, let's restructure.

    # Precompute conflict matrix
    # conflicts[i][j] = True if dot(candidates[i], candidates[j]) > 1
    
    # To speed up, we can represent vectors as bitmasks?
    # Or just use numpy matrix multiplication?
    # C is (1320, 11).
    # Dot matrix = C @ C.T
    # This is 1320x1320.
    # C is int8. C.T is int8. Result is int16/int32.
    
    C = np.array(candidates, dtype=np.int16)
    dot_matrix = C @ C.T
    
    # Conflicts where dot > 1
    # Note: dot_matrix[i][i] = 3. We don't care about self-conflict for removal logic 
    # if we remove i explicitly.
    # But for j != i, we check dot_matrix[i][j] > 1.
    
    # Greedy
    best_size = 0
    best_indices = []
    
    for restart in range(50): # Try 50 restarts
        if time.time() - start_time > deadline:
            break
            
        indices = list(range(n))
        random.shuffle(indices)
        
        selected = []
        available = np.ones(n, dtype=bool)
        
        for i in indices:
            if not available[i]:
                continue
            
            selected.append(i)
            
            # Remove i and all j where dot_matrix[i][j] > 1
            # We can use boolean masking
            # conflicts_with_i = dot_matrix[i] > 1
            # available[conflicts_with_i] = False
            
            # This is O(N) per selection. Total O(N^2).
            # 1320^2 ~ 1.7M operations.
            # 50 restarts * 1.7M = 85M operations.
            # NumPy boolean masking is fast.
            
            conflicts = dot_matrix[i] > 1
            available[conflicts] = False
            
        if len(selected) > best_size:
            best_size = len(selected)
            best_indices = selected
            
        # If we found a set larger than seed (220), we can stop early?
        # No, we want max.
        
    # Convert best_indices to vectors
    if best_indices:
        result = C[best_indices]
        return result
    else:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
