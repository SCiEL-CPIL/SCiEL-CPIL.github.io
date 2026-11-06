import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    n = 24
    # Compute LCM of 1..24
    L = 1
    for i in range(1, n + 1):
        L = math.lcm(L, i)
    
    # Values V[i] = L // i
    V = [0] * (n + 1)
    for i in range(1, n + 1):
        V[i] = L // i

    # Memoization for subset sum checks
    _subset_sum_cache = {}
    
    def subset_sum_exists(target, values_tuple):
        """Check if any subset of values sums to target.
        values_tuple is a tuple of integers.
        Returns True/False.
        """
        key = (target, values_tuple)
        if key in _subset_sum_cache:
            return _subset_sum_cache[key]
        
        if target < 0:
            _subset_sum_cache[key] = False
            return False
        if not values_tuple:
            _subset_sum_cache[key] = (target == 0)
            return target == 0
        if sum(values_tuple) < target:
            _subset_sum_cache[key] = False
            return False
        
        # Meet in the middle
        k = len(values_tuple)
        half = k // 2
        left = values_tuple[:half]
        right = values_tuple[half:]
        
        left_sums = set()
        for mask in range(1 << half):
            s = 0
            for i in range(half):
                if mask & (1 << i):
                    s += left[i]
            left_sums.add(s)
        
        # Check if any right sum completes a left sum to target
        for mask in range(1 << (k - half)):
            s = 0
            for i in range(k - half):
                if mask & (1 << i):
                    s += right[i]
            if target - s in left_sums:
                _subset_sum_cache[key] = True
                return True
        _subset_sum_cache[key] = False
        return False

    def is_valid_incremental(S_mask, v):
        """
        Check if adding v to S_mask (which is valid) keeps it valid.
        S_mask is an integer bitmask.
        v is an element (1-24).
        """
        # 1. Check if 1/v is sum of subset of S
        # S elements are those bits set in S_mask
        s_elements = tuple(i for i in range(1, n+1) if S_mask & (1 << (i-1)))
        s_values = tuple(V[i] for i in s_elements)
        
        if subset_sum_exists(V[v], s_values):
            return False
        
        # 2. Check for each x in S, if 1/x - 1/v is sum of subset of S \ {x, v}
        # i.e., V[x] - V[v] is sum of subset of S \ {x}
        for x in s_elements:
            target = V[x] - V[v]
            if target <= 0:
                continue
            # Values available: S \ {x}
            avail_values = tuple(V[i] for i in s_elements if i != x)
            if subset_sum_exists(target, avail_values):
                return False
                
        return True

    # Phase 1: Greedy to get initial solution - try multiple orderings
    best_mask = 0
    best_size = 0
    
    # Try greedy with different orderings
    orderings = [
        list(range(1, n+1)),           # 1,2,3,...,24
        list(range(n, 0, -1)),         # 24,23,...,1
        list(range(n, 0, -2)) + list(range(n-1, 0, -2)),  # evens then odds descending
        list(range(1, n+1, 2)) + list(range(2, n+1, 2)),  # odds then evens ascending
    ]
    
    for order in orderings:
        S_mask = 0
        for i in order:
            if is_valid_incremental(S_mask, i):
                S_mask |= (1 << (i-1))
        size = bin(S_mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = S_mask
    
    # Phase 2: Local search - try to add elements or swap
    def try_local_improve(mask):
        """Try to improve the mask by adding elements or swapping."""
        current_size = bin(mask).count('1')
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding each element not in mask
            for i in range(1, n+1):
                if not (mask & (1 << (i-1))):
                    if is_valid_incremental(mask, i):
                        new_mask = mask | (1 << (i-1))
                        new_size = bin(new_mask).count('1')
                        if new_size > current_size:
                            mask = new_mask
                            current_size = new_size
                            improved = True
                            break
            # Try swapping: remove one, add two
            if time.time() < deadline:
                s_elements = [i for i in range(1, n+1) if mask & (1 << (i-1))]
                for x in s_elements:
                    mask_without_x = mask & ~(1 << (x-1))
                    # Try adding two elements
                    for i in range(1, n+1):
                        if not (mask_without_x & (1 << (i-1))):
                            if is_valid_incremental(mask_without_x, i):
                                temp_mask = mask_without_x | (1 << (i-1))
                                for j in range(1, n+1):
                                    if not (temp_mask & (1 << (j-1))) and j != i:
                                        if is_valid_incremental(temp_mask, j):
                                            new_mask = temp_mask | (1 << (j-1))
                                            new_size = bin(new_mask).count('1')
                                            if new_size > current_size:
                                                mask = new_mask
                                                current_size = new_size
                                                improved = True
                                                break
                                    if improved:
                                        break
                        if improved:
                            break
                    if improved:
                        break
        return mask
    
    best_mask = try_local_improve(best_mask)
    best_size = bin(best_mask).count('1')
    
    # Phase 3: Branch and Bound
    def dfs(current_mask, candidates_mask, current_size):
        nonlocal best_mask, best_size, deadline, start_time
        
        if time.time() > deadline:
            return
        
        # Pruning
        if current_size + bin(candidates_mask).count('1') <= best_size:
            return
            
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
        
        # Pick a candidate - use heuristic: pick element that conflicts with most others first
        # For simplicity, pick the first available one
        lsb = candidates_mask & -candidates_mask
        v_idx = lsb.bit_length() # 1-based index
        v = v_idx
        
        # Branch 1: Include v
        if is_valid_incremental(current_mask, v):
            new_mask = current_mask | (1 << (v-1))
            new_candidates = candidates_mask & ~(1 << (v-1))
            dfs(new_mask, new_candidates, current_size + 1)
            
        # Branch 2: Exclude v
        new_candidates = candidates_mask & ~(1 << (v-1))
        dfs(current_mask, new_candidates, current_size)
        
    # Run B&B
    all_candidates = (1 << n) - 1
    dfs(0, all_candidates, 0)
    
    # Convert best_mask to list
    result = [i for i in range(1, n+1) if best_mask & (1 << (i-1))]
    
    return sorted(result)
