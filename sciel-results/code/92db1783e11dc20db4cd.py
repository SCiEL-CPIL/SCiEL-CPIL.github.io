"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 75  # safety margin before 90s kill
    N = 4096
    MAX_SUM = 3 * N  # 12288
    
    # Powers of 4 form a valid base set with no triple-sum collisions
    base = [4**i for i in range(10) if 4**i <= 4096]
    # 1, 4, 16, 64, 256, 1024, 4096
    
    def build_sum_array(lst):
        """Build boolean array where arr[s] = True if s is a triple sum of lst."""
        arr = np.zeros(MAX_SUM + 1, dtype=bool)
        n = len(lst)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    arr[lst[i] + lst[j] + lst[k]] = True
        return arr
    
    def can_add_fast(lst, sums_arr, candidate):
        """Check if adding candidate to lst keeps all triple sums unique using boolean array."""
        n = len(lst)
        
        # Triples with one candidate: candidate + lst[i] + lst[j] for i <= j
        for i in range(n):
            for j in range(i, n):
                s = candidate + lst[i] + lst[j]
                if s <= MAX_SUM and sums_arr[s]:
                    return False
        
        # Triples with two candidates: 2*candidate + lst[i]
        for i in range(n):
            s = 2 * candidate + lst[i]
            if s <= MAX_SUM and sums_arr[s]:
                return False
        
        # Triple with three candidates: 3*candidate
        s = 3 * candidate
        if s <= MAX_SUM and sums_arr[s]:
            return False
        
        return True
    
    def update_sums_arr(sums_arr, lst, candidate):
        """Update sums_arr to include new triple sums involving candidate."""
        n = len(lst)
        
        for i in range(n):
            for j in range(i, n):
                sums_arr[candidate + lst[i] + lst[j]] = True
        
        for i in range(n):
            sums_arr[2 * candidate + lst[i]] = True
        
        sums_arr[3 * candidate] = True
    
    # Initial sum array for base
    base_sums = build_sum_array(base)
    
    best_set = list(base)
    best_len = len(base)
    
    # Strategy 1: Randomized greedy with many restarts
    num_restarts = 200
    
    for attempt in range(num_restarts):
        if time.time() > deadline:
            break
        
        current = list(base)
        sums_arr = base_sums.copy()
        
        # Generate candidate pool
        base_set = set(base)
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for cand in candidates:
            if time.time() > deadline:
                break
            if len(current) >= 80:
                break
            if cand in base_set and cand in current:
                continue
            if can_add_fast(current, sums_arr, cand):
                current.append(cand)
                update_sums_arr(sums_arr, current[:-1], cand)
        
        if len(current) > best_len:
            best_len = len(current)
            best_set = list(current)
    
    # Strategy 2: Systematic ascending order
    if time.time() < deadline - 3 and best_len < 80:
        current = list(base)
        sums_arr = base_sums.copy()
        current_set = set(current)
        
        for cand in range(1, N + 1):
            if time.time() > deadline:
                break
            if len(current) >= 80:
                break
            if cand in current_set:
                continue
            if can_add_fast(current, sums_arr, cand):
                current.append(cand)
                current_set.add(cand)
                update_sums_arr(sums_arr, current[:-1], cand)
        
        if len(current) > best_len:
            best_len = len(current)
            best_set = list(current)
    
    # Strategy 3: Descending order
    if time.time() < deadline - 3 and best_len < 80:
        current = list(base)
        sums_arr = base_sums.copy()
        current_set = set(current)
        
        for cand in range(N, 0, -1):
            if time.time() > deadline:
                break
            if len(current) >= 80:
                break
            if cand in current_set:
                continue
            if can_add_fast(current, sums_arr, cand):
                current.append(cand)
                current_set.add(cand)
                update_sums_arr(sums_arr, current[:-1], cand)
        
        if len(current) > best_len:
            best_len = len(current)
            best_set = list(current)
    
    # Strategy 4: Local search / swap improvement on best_set
    if time.time() < deadline - 5 and best_len < 80:
        # Try to improve by removing one element and trying to add two others
        for _ in range(10):
            if time.time() > deadline:
                break
            
            trial = list(best_set)
            sums_arr = build_sum_array(trial)
            trial_set = set(trial)
            
            # Try removing each element and see if we can add more
            for idx in range(len(trial)):
                if time.time() > deadline:
                    break
                
                removed = trial[idx]
                trial2 = trial[:idx] + trial[idx+1:]
                trial2_set = set(trial2)
                
                # Rebuild sums for trial2 (expensive but only done a few times)
                sums2 = build_sum_array(trial2)
                
                added = 0
                for cand in range(1, N + 1):
                    if time.time() > deadline:
                        break
                    if len(trial2) + added >= len(trial) + 1:
                        break
                    if cand in trial2_set:
                        continue
                    if can_add_fast(trial2, sums2, cand):
                        trial2.append(cand)
                        trial2_set.add(cand)
                        update_sums_arr(sums2, trial2[:-1], cand)
                        added += 1
                
                if len(trial2) > len(trial):
                    trial = trial2
                    trial_set = set(trial)
                    break
            
            if len(trial) > best_len:
                best_len = len(trial)
                best_set = list(trial)
    
    return best_set[:80]
# EVOLVE-BLOCK-END
