# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 580  # Safety margin
    
    # Generate candidate vectors
    # We consider vectors with entries in {-1, 0, 1} and weight up to 4
    # And vectors with one entry +-2 and rest 0 (norm 4)
    # And vectors with one entry +-2 and one entry +-1 (norm 5)
    
    candidates = []
    
    # Weight 1, 2, 3, 4 with entries in {-1, 1}
    for weight in range(1, 5):
        for indices in itertools.combinations(range(d), weight):
            for signs in itertools.product([-1, 1], repeat=weight):
                v = np.zeros(d, dtype=np.int64)
                for idx, s in zip(indices, signs):
                    v[idx] = s
                candidates.append(v)
                
    # Norm 4 vectors: one +-2
    for i in range(d):
        for s in [-1, 1]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = 2 * s
            candidates.append(v)
            
    # Norm 5 vectors: one +-2, one +-1
    for i in range(d):
        for j in range(d):
            if i == j: continue
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = 2 * si
                    v[j] = sj
                    candidates.append(v)
                    
    # Norm 8 vectors: one +-2, two +-1? No, let's stick to small norms first.
    # Maybe add vectors with two +-2? Norm 8.
    for i in range(d):
        for j in range(d):
            if i == j: continue
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    candidates.append(v)

    # Remove duplicates
    candidates = np.array(candidates)
    candidates = np.unique(candidates, axis=0)
    
    # Precompute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    best_set = []
    best_size = 0
    
    num_candidates = len(candidates)
    
    # Greedy search with restarts
    # We try different random orderings
    # To keep it fast, we limit the number of restarts or time per restart
    
    max_restarts = 500
    restarts_done = 0
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random shuffle
        order = np.random.permutation(num_candidates)
        
        current_set = []
        current_norms_sq = []
        max_norm_sq = 0
        
        for idx in order:
            v = candidates[idx]
            v_norm_sq = norms_sq[idx]
            
            new_max_norm_sq = max(max_norm_sq, v_norm_sq)
            
            # Check if v can be added
            # Condition 1: dist(v, u) > new_max_norm_sq for all u in current_set
            # Condition 2: dist(u, w) > new_max_norm_sq for all u, w in current_set
            
            # Check Condition 2 first if new_max_norm_sq > max_norm_sq
            if new_max_norm_sq > max_norm_sq:
                valid = True
                # Check all pairs in current_set
                for i in range(len(current_set)):
                    u = current_set[i]
                    for j in range(i+1, len(current_set)):
                        w = current_set[j]
                        dist_sq = np.sum((u - w) ** 2)
                        if dist_sq <= new_max_norm_sq:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
            
            # Check Condition 1
            valid = True
            for u in current_set:
                dist_sq = np.sum((v - u) ** 2)
                if dist_sq <= new_max_norm_sq:
                    valid = False
                    break
            
            if valid:
                current_set.append(v)
                max_norm_sq = new_max_norm_sq
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
        # Heuristic: if we found a large set, we might break early?
        # But let's try to improve.
        
    if best_size == 0:
        # Fallback to trivial solution
        points = np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
    else:
        points = np.array(best_set).astype(np.int64)
        
    return points

# EVOLVE-BLOCK-END
