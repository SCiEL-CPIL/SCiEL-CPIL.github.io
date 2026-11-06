import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    max_val = 4096
    max_sum = 3 * max_val + 1
    
    def try_add(x, selected, S, used_sums):
        """Try to add x to the current set. Returns (success, new_S, new_used_sums) if successful."""
        if len(selected) == 0:
            S_new = np.array([x], dtype=np.int32)
            used_sums_new = used_sums.copy()
            used_sums_new[3*x] = True
            return True, S_new, used_sums_new
        
        k = len(selected)
        
        # Generate new triple sums involving x
        s1 = np.array([3*x], dtype=np.int32)
        s2 = 2*x + S
        i_idx, j_idx = np.triu_indices(k)
        s3 = x + S[i_idx] + S[j_idx]
        
        all_new_sums = np.concatenate((s1, s2, s3))
        
        # Check for duplicates within new sums
        if len(np.unique(all_new_sums)) != len(all_new_sums):
            return False, None, None
            
        # Check against used_sums
        if np.any(used_sums[all_new_sums]):
            return False, None, None
            
        # If valid, update
        S_new = np.append(S, x)
        used_sums_new = used_sums.copy()
        used_sums_new[all_new_sums] = True
        return True, S_new, used_sums_new

    def greedy_build(start_from=1, skip_pattern=None):
        """Build a set using greedy strategy."""
        selected = []
        S = np.array([], dtype=np.int32)
        used_sums = np.zeros(max_sum, dtype=bool)
        
        for x in range(start_from, max_val + 1):
            if time.time() > deadline:
                break
            if skip_pattern is not None and skip_pattern(x):
                continue
            success, S_new, used_sums_new = try_add(x, selected, S, used_sums)
            if success:
                selected.append(x)
                S = S_new
                used_sums = used_sums_new
        
        return selected, S, used_sums

    # Primary strategy: Greedy from 1
    best_set, best_S, best_used = greedy_build(start_from=1)
    best_len = len(best_set)
    
    elapsed = time.time() - start_time
    
    # Local improvement: try to swap elements or add skipped ones
    if elapsed < 40.0:
        # Try local search: for each element in the set, try replacing it with a larger number
        # that was skipped, to see if we can then add more elements
        improved = False
        for _ in range(3):  # Multiple passes
            if time.time() > deadline - 5:
                break
            
            current_set = list(best_set)
            current_S = best_S.copy()
            current_used = best_used.copy()
            
            # Try to insert elements that were skipped
            skipped = [x for x in range(1, max_val + 1) if x not in set(current_set)]
            
            # Try each skipped element to see if it can be added
            for x in skipped:
                if time.time() > deadline - 5:
                    break
                success, S_new, used_new = try_add(x, current_set, current_S, current_used)
                if success:
                    current_set.append(x)
                    current_S = S_new
                    current_used = used_new
                    improved = True
                    if len(current_set) > best_len:
                        best_set = list(current_set)
                        best_S = current_S
                        best_used = current_used
                        best_len = len(best_set)
            
            if not improved:
                break
    
    # Try different greedy strategies with slight variations
    if time.time() < deadline - 10:
        # Try greedy starting from different offsets
        for offset in [2, 3, 5]:
            if time.time() > deadline - 5:
                break
            alt_set, alt_S, alt_used = greedy_build(start_from=1)
            # Actually, let's try a different approach: skip every k-th small number
            selected = []
            S = np.array([], dtype=np.int32)
            used_sums = np.zeros(max_sum, dtype=bool)
            
            # Modified greedy: skip some small numbers to see if we get a larger set
            skip_small = set(range(1, offset + 1))
            for x in range(1, max_val + 1):
                if time.time() > deadline:
                    break
                if x in skip_small:
                    continue
                success, S_new, used_sums_new = try_add(x, selected, S, used_sums)
                if success:
                    selected.append(x)
                    S = S_new
                    used_sums = used_sums_new
            
            if len(selected) > best_len:
                best_set = selected
                best_len = len(selected)
    
    return best_set
