"""Program for B_3[1] set construction with unique triple sums."""

import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    n_max = 4096
    max_sum = 3 * n_max
    
    # Helper to check and add x to a set represented by list and sum array
    def try_add(x, selected, sums_arr):
        # 1. (x, x, x)
        s = 3 * x
        if s <= max_sum and sums_arr[s]:
            return False
        new_sums = [s]
        
        # 2. (y, x, x) -> y + 2x
        # 3. (y, y, x) -> 2y + x
        for y in selected:
            s1 = y + 2 * x
            if s1 <= max_sum and sums_arr[s1]:
                return False
            new_sums.append(s1)
            
            s2 = 2 * y + x
            if s2 <= max_sum and sums_arr[s2]:
                return False
            new_sums.append(s2)
            
        # 4. (y, z, x) with y < z in selected
        n_sel = len(selected)
        for i in range(n_sel):
            y = selected[i]
            for j in range(i + 1, n_sel):
                z = selected[j]
                s = y + z + x
                if s <= max_sum and sums_arr[s]:
                    return False
                new_sums.append(s)
                
        # If all good, commit
        for s in new_sums:
            if s <= max_sum:
                sums_arr[s] = True
        selected.append(x)
        return True

    def greedy_pass(order, initial_selected=None, initial_sums=None):
        if initial_selected is not None:
            selected = list(initial_selected)
            if initial_sums is not None:
                sums_arr = initial_sums.copy()
            else:
                sums_arr = np.zeros(max_sum + 1, dtype=bool)
                for i in range(len(selected)):
                    for j in range(i, len(selected)):
                        for k in range(j, len(selected)):
                            s = selected[i] + selected[j] + selected[k]
                            if s <= max_sum:
                                sums_arr[s] = True
        else:
            selected = []
            sums_arr = np.zeros(max_sum + 1, dtype=bool)
            
        for x in order:
            if time.time() > deadline:
                break
            if try_add(x, selected, sums_arr):
                pass
        return selected

    best_set = []
    
    # Strategy 1: Random Greedy (start from scratch)
    # Strategy 2: Skeleton Greedy (start with powers of 4)
    
    # Skeleton: Powers of 4
    skeleton = [4**i for i in range(7) if 4**i <= n_max]
    # skeleton is [1, 4, 16, 64, 256, 1024, 4096]
    
    # We can also try other skeletons, e.g., scaled powers of 2 that are valid?
    # But we know powers of 2 are invalid.
    # Let's just use random and skeleton.
    
    # Pre-shuffle candidates for skeleton passes
    all_candidates = list(range(1, n_max + 1))
    remaining = [x for x in range(1, n_max + 1) if x not in set(skeleton)]
    
    while time.time() < deadline:
        # Random Pass
        candidates = list(range(1, n_max + 1))
        random.shuffle(candidates)
        res = greedy_pass(candidates)
        if len(res) > len(best_set):
            best_set = res
            
        if time.time() > deadline:
            break
        
        # Skeleton Pass
        # Start with skeleton, then try to add others
        # We need to initialize sums_arr with skeleton sums
        selected = list(skeleton)
        sums_arr = np.zeros(max_sum + 1, dtype=bool)
        
        # Add skeleton sums
        for i in range(len(selected)):
            for j in range(i, len(selected)):
                for k in range(j, len(selected)):
                    s = selected[i] + selected[j] + selected[k]
                    if s <= max_sum:
                        sums_arr[s] = True
        
        # Candidates to try: all numbers not in skeleton
        # Shuffle them
        remaining_shuffled = remaining.copy()
        random.shuffle(remaining_shuffled)
        
        for x in remaining_shuffled:
            if time.time() > deadline:
                break
            if try_add(x, selected, sums_arr):
                pass
                
        if len(selected) > len(best_set):
            best_set = selected
            
    return best_set
