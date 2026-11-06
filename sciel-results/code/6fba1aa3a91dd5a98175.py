import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 88.0
    
    n = 256
    # Safe base set: (2n/3, n] = [171, 256]
    base_start = 171
    S = set(range(base_start, n + 1))
    S_arr = np.array(sorted(S), dtype=np.int64)
    
    def can_add_fast(a, S_set):
        """Check if a can be added to S_set without violating the property."""
        # For any a < b <= c in S_set, a must not divide b+c
        # This means for each b in S_set with b > a, we need no c in S_set
        # with c >= b such that (b+c) % a == 0
        # i.e., c % a == (-b) % a and c >= b
        
        elems = [x for x in S_set if x > a]
        if len(elems) < 2:
            return True
        
        # Group by remainder mod a
        rem_groups = {}
        for e in elems:
            r = e % a
            if r not in rem_groups:
                rem_groups[r] = []
            rem_groups[r].append(e)
        
        # Sort each group for binary search
        for r in rem_groups:
            rem_groups[r].sort()
        
        for b in elems:
            target_rem = (-b) % a
            if target_rem in rem_groups:
                lst = rem_groups[target_rem]
                # Find first element > b (or >= b, since b <= c is allowed)
                idx = np.searchsorted(lst, b, side='left')
                if idx < len(lst):
                    return False
        return True
    
    def can_add_vectorized(a, S_arr):
        """Vectorized check for whether a can be added."""
        mask = S_arr > a
        elems = S_arr[mask]
        if len(elems) < 2:
            return True
        
        # For each b in elems, check if there's c in elems with c >= b and (b+c)%a == 0
        # (b+c) % a == 0 means c % a == (-b) % a
        # So for each b, we need to check if there exists c in elems with c >= b and c % a == (-b)%a
        
        b_rems = elems % a
        target_rems = (-b_rems) % a
        c_rems = elems % a
        
        # For each b (index i in elems), check if any c (index j in elems) has:
        # elems[j] >= elems[i] and c_rems[j] == target_rems[i]
        
        # This is O(n^2) in the worst case but vectorized
        # Create a matrix: can_pair[i,j] = (elems[j] >= elems[i]) and (c_rems[j] == target_rems[i])
        # We just need to know if any True exists
        
        # More efficient: for each b, check if max c with c % a == target_rem[b] >= b
        # Group elems by remainder
        unique_rems, inverse = np.unique(c_rems, return_inverse=True)
        
        for i in range(len(elems)):
            tr = target_rems[i]
            # Find all indices where c_rems == tr
            mask_c = c_rems == tr
            if not np.any(mask_c):
                continue
            c_vals = elems[mask_c]
            if len(c_vals) > 0 and c_vals.max() >= elems[i]:
                return False
        return True
    
    def try_add(a, S_set, S_arr):
        if a in S_set:
            return False, S_set, S_arr
        if can_add_fast(a, S_set):
            S_set.add(a)
            S_arr = np.append(S_arr, a)
            S_arr.sort()
            return True, S_set, S_arr
        return False, S_set, S_arr
    
    # Phase 1: Greedy from top to bottom
    for a in range(base_start - 1, 0, -1):
        if time.time() > deadline - 2:
            break
        try_add(a, S, S_arr)
    
    # Phase 2: Randomized passes to add remaining elements
    remaining = [a for a in range(1, base_start) if a not in S]
    
    for _ in range(8):
        if time.time() > deadline - 5:
            break
        random.shuffle(remaining)
        for a in remaining:
            if time.time() > deadline - 2:
                break
            if a in S:
                continue
            if can_add_fast(a, S):
                S.add(a)
                S_arr = np.append(S_arr, a)
                S_arr.sort()
                remaining.remove(a)
        if not remaining:
            break
    
    # Phase 3: Swap-based local search
    # Try removing one element and adding one or two others
    if time.time() < deadline - 10:
        S_list = list(S)
        max_len = len(S)
        
        for iteration in range(300):
            if time.time() > deadline - 3:
                break
            
            # Try removing a random element
            i = random.randint(0, len(S_list) - 1)
            elem_to_remove = S_list[i]
            
            S_temp = S - {elem_to_remove}
            S_temp_list = [x for x in S_list if x != elem_to_remove]
            
            # Try to add elements that were previously blocked
            candidates = [a for a in range(1, n + 1) if a not in S_temp and a != elem_to_remove]
            random.shuffle(candidates)
            
            added_count = 0
            for a in candidates:
                if time.time() > deadline - 3:
                    break
                if can_add_fast(a, S_temp):
                    S_temp.add(a)
                    added_count += 1
                    if added_count >= 2:
                        break
            
            if added_count >= 1 and len(S_temp) > max_len:
                S = S_temp
                S_list = list(S)
                max_len = len(S)
                S_arr = np.array(sorted(S), dtype=np.int64)
    
    # Phase 4: Final polish
    if time.time() < deadline - 2:
        remaining_final = [a for a in range(1, n + 1) if a not in S]
        random.shuffle(remaining_final)
        for a in remaining_final:
            if time.time() > deadline - 1:
                break
            if can_add_fast(a, S):
                S.add(a)
    
    return list(S)
