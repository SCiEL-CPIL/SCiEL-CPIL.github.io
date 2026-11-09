import numpy as np
from math import gcd
import time

def construct():
    N = 512
    size = 511
    
    # Precompute conflict matrix
    conflict = np.zeros((size, size), dtype=bool)
    for i in range(size):
        a = i + 1
        for j in range(i + 1, size):
            b = j + 1
            if gcd(a, b) > 1:
                conflict[i][j] = True
                conflict[j][i] = True
    
    # Weights
    weights = np.array([1.0 / (N - (i + 1)) for i in range(size)])
    
    # Greedy Initialization
    chosen_mask = np.zeros(size, dtype=bool)
    # Sort indices by weight descending
    order = np.argsort(weights)[::-1]
    for i in order:
        if not np.any(conflict[i] & chosen_mask):
            chosen_mask[i] = True
            
    # Local Search
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < 100:
        improved = False
        
        # 1. Try to add any missing compatible node
        # Find nodes not chosen and not conflicting with chosen
        # conflict_with_chosen[i] is True if i conflicts with ANY chosen node
        conflict_with_chosen = np.any(conflict[:, chosen_mask], axis=1)
        candidates = ~chosen_mask & ~conflict_with_chosen
        if np.any(candidates):
            # Pick the one with max weight
            cand_indices = np.where(candidates)[0]
            best_idx = cand_indices[np.argmax(weights[cand_indices])]
            chosen_mask[best_idx] = True
            improved = True
            continue # Restart to check for more additions or other moves
            
        # 2. Try 1-for-1 swap
        chosen_indices = np.where(chosen_mask)[0]
        swap_done = False
        for a_idx in chosen_indices:
            # Temporarily remove a
            temp_mask = chosen_mask.copy()
            temp_mask[a_idx] = False
            
            # Find candidates for replacement
            conflict_with_temp = np.any(conflict[:, temp_mask], axis=1)
            candidates = ~temp_mask & ~conflict_with_temp
            if np.any(candidates):
                cand_indices = np.where(candidates)[0]
                # Exclude a_idx from consideration (it's already excluded by ~temp_mask)
                # Find best candidate
                best_b_idx = cand_indices[np.argmax(weights[cand_indices])]
                if weights[best_b_idx] > weights[a_idx]:
                    chosen_mask[a_idx] = False
                    chosen_mask[best_b_idx] = True
                    improved = True
                    swap_done = True
                    break # Restart
        if swap_done:
            continue
            
        # 3. Try 1-for-many removal (drop a, add optimal set of new nodes)
        removal_done = False
        for a_idx in chosen_indices:
            temp_mask = chosen_mask.copy()
            temp_mask[a_idx] = False
            
            conflict_with_temp = np.any(conflict[:, temp_mask], axis=1)
            candidates = ~temp_mask & ~conflict_with_temp
            if np.any(candidates):
                # Greedily select best subset from candidates
                cand_indices = np.where(candidates)[0]
                cand_order = cand_indices[np.argsort(weights[cand_indices])[::-1]]
                
                new_mask = temp_mask.copy()
                gain = 0.0
                for b_idx in cand_order:
                    if not np.any(conflict[b_idx] & new_mask):
                        new_mask[b_idx] = True
                        gain += weights[b_idx]
                        
                if gain > weights[a_idx]:
                    # Apply change
                    chosen_mask = new_mask
                    improved = True
                    removal_done = True
                    break # Restart
        if removal_done:
            continue

    return [int(i + 1) for i in np.where(chosen_mask)[0]]
