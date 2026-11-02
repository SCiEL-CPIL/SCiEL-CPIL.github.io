import numpy as np
import itertools
import random

D = 11

def construct():
    # Generate candidate vectors
    # We focus on integer vectors with small norms to maximize packing density
    # Norm squared 2: +/- e_i +/- e_j
    # Norm squared 4: +/- 2 e_i, +/- e_i +/- e_j +/- e_k +/- e_l (4 ones)
    
    candidates = []
    
    # Add S2 vectors (Norm sq 2)
    for i in range(D):
        for j in range(D):
            if i != j:
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = s1
                        v[j] = s2
                        candidates.append(v)
                        
    # Add Norm sq 4 vectors: 2*e_i
    for i in range(D):
        for s in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            candidates.append(v)
            
    # Add Norm sq 4 vectors: 4 ones
    # This might be too many, but let's check count: C(11,4) * 16 = 330 * 16 = 5280. Manageable.
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    
    best_set = []
    best_score = 0
    
    # Greedy search with randomization
    # We try multiple random orders
    for trial in range(50):
        random.seed(trial)
        indices = np.arange(len(candidates))
        random.shuffle(indices)
        
        current_set = []
        current_max_norm_sq = 0
        
        for idx in indices:
            v = candidates[idx]
            v_norm_sq = np.dot(v, v)
            
            # If this vector's norm is larger than current max, we must ensure
            # it is far enough from all existing vectors.
            # Condition: min_dist_sq >= max(current_max_norm_sq, v_norm_sq)
            
            if not current_set:
                current_set.append(v)
                current_max_norm_sq = v_norm_sq
                continue
                
            # Check distance to all vectors in current_set
            # Optimize: if v_norm_sq > current_max_norm_sq, threshold is v_norm_sq
            # else threshold is current_max_norm_sq
            
            threshold = max(current_max_norm_sq, v_norm_sq)
            
            valid = True
            for u in current_set:
                diff = v - u
                dist_sq = np.dot(diff, diff)
                if dist_sq < threshold:
                    valid = False
                    break
            
            if valid:
                current_set.append(v)
                if v_norm_sq > current_max_norm_sq:
                    current_max_norm_sq = v_norm_sq
                    
        if len(current_set) > best_score:
            best_score = len(current_set)
            best_set = current_set
            
    return np.array(best_set, dtype=np.int64)
