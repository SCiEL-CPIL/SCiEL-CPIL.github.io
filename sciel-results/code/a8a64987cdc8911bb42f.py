import numpy as np
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin
    
    max_val = 4096
    max_sum = 3 * max_val + 1
    
    # Helper function to check if x can be added to S
    def can_add(x, S, used_sums):
        if not S:
            return True, np.array([3*x], dtype=np.int32)
        
        k = len(S)
        # 1. x + x + x
        s1 = 3 * x
        
        # 2. x + x + s_i
        s2 = 2 * x + S
        
        # 3. x + s_i + s_j (i <= j)
        i_idx, j_idx = np.triu_indices(k)
        s3 = x + S[i_idx] + S[j_idx]
        
        all_new_sums = np.concatenate(([s1], s2, s3))
        
        # 1. Check collisions with used_sums
        if np.any(used_sums[all_new_sums]):
            return False, None
            
        # 2. Check internal collisions
        if len(np.unique(all_new_sums)) != len(all_new_sums):
            return False, None
            
        return True, all_new_sums

    # Phase 1: Deterministic Greedy
    used_sums = np.zeros(max_sum, dtype=bool)
    selected = []
    S = np.array([], dtype=np.int32)
    
    for x in range(1, max_val + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        valid, new_sums = can_add(x, S, used_sums)
        if valid:
            selected.append(x)
            S = np.append(S, x)
            used_sums[new_sums] = True

    best_set = selected[:]
    best_S = S.copy()
    best_used = used_sums.copy()
    best_size = len(best_set)
    
    # Phase 2: Randomized Greedy Restarts
    attempts = 0
    
    while time.time() - start_time < DEADLINE - 2.0:
        attempts += 1
        
        # Shuffle candidates
        candidates = np.random.permutation(range(1, max_val + 1))
        
        current_set = []
        current_S = np.array([], dtype=np.int32)
        current_used = np.zeros(max_sum, dtype=bool)
        
        # Optimization: Early exit if we can't beat best_size
        for idx, x in enumerate(candidates):
            remaining = len(candidates) - idx
            if len(current_set) + remaining <= best_size:
                break
                
            valid, new_sums = can_add(x, current_S, current_used)
            if valid:
                current_set.append(x)
                current_S = np.append(current_S, x)
                current_used[new_sums] = True
        
        if len(current_set) > best_size:
            best_set = current_set
            best_S = current_S
            best_used = current_used
            best_size = len(best_set)

    return best_set
