import time
import random
import numpy as np

def construct():
    """
    Constructs a B_3[1] set (unique triple sums) of maximum size.
    
    Strategy:
    1. Use a known-valid sparse skeleton (powers of 4) to guarantee a base set.
    2. Use fast incremental conflict checking (O(k^2) per insertion) to fill gaps.
    3. Perform multiple randomized restarts to find larger sets than a single greedy pass.
    4. Track the best set found across all restarts.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s kill
    
    n_max = 4096
    max_sum = 3 * n_max
    
    # --- Fast Incremental Conflict Checking ---
    
    def try_add(x, selected, sums_arr):
        """
        Check if x can be added to selected without violating B_3[1] property.
        Returns True if successful, and modifies selected/sums_arr in place.
        """
        # 1. (x, x, x)
        s = 3 * x
        if sums_arr[s]:
            return False
        new_sums = [s]
        
        # 2. (y, x, x) -> y + 2x  and  (y, y, x) -> 2y + x
        for y in selected:
            s1 = y + 2 * x
            if sums_arr[s1]:
                return False
            new_sums.append(s1)
            
            s2 = 2 * y + x
            if sums_arr[s2]:
                return False
            new_sums.append(s2)
                
        # 3. (y, z, x) with y < z in selected
        n_sel = len(selected)
        for i in range(n_sel):
            y = selected[i]
            for j in range(i + 1, n_sel):
                z = selected[j]
                s = y + z + x
                if sums_arr[s]:
                    return False
                new_sums.append(s)
                
        # If all good, commit
        for s in new_sums:
            sums_arr[s] = True
        selected.append(x)
        return True

    def build_sums_array(S):
        """Build a boolean array of all triple sums for a given set S."""
        sums_arr = np.zeros(max_sum + 1, dtype=bool)
        k = len(S)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    sums_arr[S[i] + S[j] + S[l]] = True
        return sums_arr

    def greedy_pass(order, start_list=None):
        """
        Perform a greedy pass adding elements in the given order.
        If start_list is provided, initialize with it.
        """
        if start_list:
            selected = list(start_list)
            sums_arr = build_sums_array(selected)
            existing = set(selected)
        else:
            selected = []
            sums_arr = np.zeros(max_sum + 1, dtype=bool)
            existing = set()
            
        for x in order:
            if time.time() > deadline:
                break
            if x in existing:
                continue
            if try_add(x, selected, sums_arr):
                existing.add(x)
                
        return selected

    # --- Main Search Loop ---
    
    best_set = []
    
    # Skeleton: Powers of 4 are valid B_3[1] because 1+1+1=3 < 4 (no carries)
    skeleton = [4**i for i in range(7) if 4**i <= n_max]
    # skeleton = [1, 4, 16, 64, 256, 1024, 4096]
    
    # Try a few different strategies:
    # 1. Pure random greedy
    # 2. Skeleton-based greedy (start with skeleton, fill gaps)
    # 3. Forward/Backward greedy
    
    strategies = 0
    while time.time() < deadline:
        strategies += 1
        
        # Strategy 1: Random Greedy
        candidates = list(range(1, n_max + 1))
        random.shuffle(candidates)
        res = greedy_pass(candidates)
        if len(res) > len(best_set):
            best_set = res
        
        if time.time() > deadline - 10:  # Need more time for next strategy
            break
            
        # Strategy 2: Skeleton-based Greedy
        # Start with skeleton, then try to add remaining numbers in random order
        remaining = [x for x in range(1, n_max + 1) if x not in set(skeleton)]
        random.shuffle(remaining)
        res = greedy_pass(remaining, start_list=skeleton)
        if len(res) > len(best_set):
            best_set = res
        
        if time.time() > deadline - 10:
            break
            
        # Strategy 3: Forward Greedy (1 to 4096)
        forward_order = list(range(1, n_max + 1))
        res = greedy_pass(forward_order)
        if len(res) > len(best_set):
            best_set = res
            
        if time.time() > deadline - 10:
            break
        
        # Strategy 4: Backward Greedy (4096 to 1)
        backward_order = list(range(n_max, 0, -1))
        res = greedy_pass(backward_order)
        if len(res) > len(best_set):
            best_set = res
        
        # If we've done several rounds, increase randomness
        if strategies > 4:
            # Mix: start with random subset of skeleton, then fill
            partial_skeleton = skeleton[:random.randint(3, 6)]
            remaining = [x for x in range(1, n_max + 1) if x not in set(partial_skeleton)]
            random.shuffle(remaining)
            res = greedy_pass(remaining, start_list=partial_skeleton)
            if len(res) > len(best_set):
                best_set = res

    # If we found a good set, return it
    if best_set:
        return best_set
    
    # Fallback: Just return the skeleton (shouldn't happen)
    return skeleton

# Quick self-test
if __name__ == "__main__":
    result = construct()
    print(f"Found set of size: {len(result)}")
    print(f"Min: {min(result)}, Max: {max(result)}")
    
    # Verify
    n = len(result)
    sums_seen = set()
    valid = True
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                s = result[i] + result[j] + result[k]
                if s in sums_seen:
                    valid = False
                    break
                sums_seen.add(s)
            if not valid:
                break
        if not valid:
            break
    print(f"Valid B_3[1] set: {valid}")
