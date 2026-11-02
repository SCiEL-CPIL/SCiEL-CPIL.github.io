"""Greedy Packing on Norm-Squared 3 Lattice Points."""
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    D = 11
    # Generate all integer vectors with exactly three non-zero entries, each +/-1.
    # Norm squared = 3. Norm = sqrt(3).
    # Condition: min dist >= sqrt(3).
    # dist^2 = 6 - 2 dot.
    # 6 - 2 dot >= 3 => 2 dot <= 3 => dot <= 1.
    
    candidates = []
    indices = range(D)
    # Choose 3 distinct indices
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                # 8 sign combinations
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        for s3 in [1, -1]:
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = s1
                            v[j] = s2
                            v[k] = s3
                            candidates.append(v)
                            
    candidates = np.array(candidates)
    N = len(candidates)
    
    # Precompute dot products matrix? 1320x1320 is ~1.7M floats. ~14MB. OK.
    # But we only need to check conflicts greedily.
    
    best_set = []
    best_score = 0
    
    # Greedy with random restarts
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Random shuffle indices
        idx = np.arange(N)
        np.random.shuffle(idx)
        
        selected = []
        selected_vectors = []
        
        for i in idx:
            if time.time() > deadline - 2:
                break
                
            v = candidates[i]
            # Check against all selected vectors
            # Dot product must be <= 1
            conflict = False
            for w in selected_vectors:
                dot = np.dot(v, w)
                if dot > 1:
                    conflict = True
                    break
                    
            if not conflict:
                selected_vectors.append(v)
                selected.append(i)
                
        if len(selected) > best_score:
            best_score = len(selected)
            best_set = selected
            # Optional: stop early if we find a very large set?
            # Kissing number for d=11 is likely > 300.
            if best_score > 400:
                break
                
    if not best_set:
        # Fallback to E8xA3 if greedy fails or times out badly
        # But greedy should find at least some.
        # Let's just return the best found.
        return np.array(best_set, dtype=np.int64)
        
    result = candidates[best_set]
    return result
