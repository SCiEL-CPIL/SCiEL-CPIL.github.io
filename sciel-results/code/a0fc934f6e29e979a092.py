"""Improved program for large integer sets with unique triple sums."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    N = 4096
    MAX_SUM = 3 * N  # 12288
    
    # Use a boolean array for O(1) triple-sum lookups
    # sums[i] is True if sum i has been used
    sums_arr = np.zeros(MAX_SUM + 1, dtype=bool)
    
    def init_state(lst):
        """Initialize the sums array for a given list."""
        sums_arr[:] = False
        n = len(lst)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    s = lst[i] + lst[j] + lst[k]
                    sums_arr[s] = True
        return sums_arr.copy()
    
    def try_add(lst, candidate, sums):
        """
        Try to add candidate to lst. Returns True if valid, else False.
        Modifies sums in place if valid.
        """
        n = len(lst)
        # Collect new sums
        new_sums = []
        
        # 1. Triples with one candidate: candidate + lst[i] + lst[j] for i <= j
        for i in range(n):
            li = lst[i]
            for j in range(i, n):
                lj = lst[j]
                s = candidate + li + lj
                if sums[s] or s in new_sums:
                    return False
                new_sums.append(s)
        
        # 2. Triples with two candidates: 2*candidate + lst[i]
        for i in range(n):
            s = 2 * candidate + lst[i]
            if sums[s] or s in new_sums:
                return False
            new_sums.append(s)
        
        # 3. Triple with three candidates: 3*candidate
        s = 3 * candidate
        if sums[s] or s in new_sums:
            return False
        new_sums.append(s)
        
        # All checks passed, update sums
        for s in new_sums:
            sums[s] = True
        return True
    
    # Base set: Powers of 4 are a strong start
    base = [4**i for i in range(10) if 4**i <= 4096]
    base_set = set(base)
    
    best_set = list(base)
    best_len = len(base)
    
    # Candidate pool
    all_candidates = [x for x in range(1, N + 1) if x not in base_set]
    
    # Randomized restarts with greedy insertion
    num_restarts = 500
    
    for attempt in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Start with base
        current_lst = list(base)
        sums = init_state(current_lst)
        
        # Shuffle candidates
        candidates = all_candidates[:]
        random.shuffle(candidates)
        
        for cand in candidates:
            if time.time() > deadline:
                break
            if len(current_lst) >= 80:
                break
            
            # Try to add candidate
            if try_add(current_lst, cand, sums):
                current_lst.append(cand)
        
        if len(current_lst) > best_len:
            best_len = len(current_lst)
            best_set = list(current_lst)
            
            # If we reached a high number, we can stop early
            if best_len >= 50:
                break
    
    return best_set[:80]
# EVOLVE-BLOCK-END
