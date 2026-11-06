import sys
from functools import lru_cache

def construct():
    max_target = 128
    
    # Generate smooth numbers up to max_target
    smooth = []
    for n in range(1, max_target + 1):
        x = n
        while x % 2 == 0:
            x //= 2
        while x % 3 == 0:
            x //= 3
        while x % 5 == 0:
            x //= 5
        if x == 1:
            smooth.append(n)
    
    N = len(smooth)
    
    # Precompute multiples mask for each smooth number
    # mult_mask[i] is a bitmask where bit j is set if smooth[j] is a multiple of smooth[i]
    mult_mask = [0] * N
    for i in range(N):
        si = smooth[i]
        mask = 0
        for j in range(N):
            if smooth[j] % si == 0:
                mask |= (1 << j)
        mult_mask[i] = mask

    results = [None] * (max_target + 1)

    # We will solve for each target using a recursive DP
    # To speed up, we can use a global memoization cache for the subproblems
    # The state is (i, sum, mask)
    # The value returned is the minimum number of additional terms needed to reach
    # a sum that is a multiple of... wait, we need to reach exactly `target`.
    # So the target is part of the problem definition, not the state, unless we
    # restructure.
    
    # Since targets are small, we can just run the DP for each target.
    # However, the subproblems (i, current_sum, mask) are the same for all targets,
    # except for the stopping condition.
    # We can define dfs(i, sum, mask) to return a dictionary: {target: min_terms}
    # This might be complex.
    
    # Simpler: Just run lru_cache for each target.
    # But lru_cache is function-local. We can define an inner function.
    
    for target in range(1, max_target + 1):
        
        @lru_cache(maxsize=None)
        def dfs(i, current_sum, mask):
            # Returns min number of terms to add to current_sum to reach target
            # using smooth numbers from index i onwards, respecting mask.
            # Returns inf if impossible.
            
            if current_sum == target:
                return 0
            
            if current_sum > target:
                return float('inf')
            
            if i == N:
                return float('inf')
            
            # Option 1: Skip smooth[i]
            res1 = dfs(i + 1, current_sum, mask)
            
            # Option 2: Take smooth[i]
            # Check if smooth[i] is allowed (not forbidden by previous picks)
            # smooth[i] is forbidden if bit i is set in mask
            if (mask & (1 << i)) == 0:
                new_mask = mask | mult_mask[i]
                next_sum = current_sum + smooth[i]
                # Pruning: if next_sum > target, skip
                if next_sum <= target:
                    sub_res = dfs(i + 1, next_sum, new_mask)
                    if sub_res < float('inf'):
                        res2 = 1 + sub_res
                    else:
                        res2 = float('inf')
                else:
                    res2 = float('inf')
            else:
                res2 = float('inf')
                
            return min(res1, res2)

        min_terms = dfs(0, 0, 0)
        
        if min_terms == float('inf'):
            # Should not happen for 1..128 with 2,3,5 smooths
            # But just in case, fallback or error
            raise ValueError(f"No antichain representation found for {target}")
        
        # Now we need to reconstruct the actual terms
        # We re-run the search to store the choices
        def find_path(i, current_sum, mask):
            if current_sum == target:
                return []
            
            if i == N:
                return None
            
            # Try skip
            # We need to know if skip leads to a solution with min_terms
            # We can just follow the greedy choice that matches the min_terms value
            
            # Option 1: Skip
            val_skip = dfs(i + 1, current_sum, mask)
            
            # Option 2: Take
            val_take = float('inf')
            if (mask & (1 << i)) == 0:
                next_sum = current_sum + smooth[i]
                if next_sum <= target:
                    sub_val = dfs(i + 1, next_sum, mask | mult_mask[i])
                    if sub_val < float('inf'):
                        val_take = 1 + sub_val
            
            # Choose the path that gives the minimum
            if val_skip <= val_take:
                # Skip
                path = find_path(i + 1, current_sum, mask)
                if path is None:
                    return None
                return path
            else:
                # Take
                new_mask = mask | mult_mask[i]
                next_sum = current_sum + smooth[i]
                path = find_path(i + 1, next_sum, new_mask)
                if path is None:
                    return None
                return [smooth[i]] + path

        path = find_path(0, 0, 0)
        if path is None:
             # This can happen if there's a tie or bug in reconstruction
             # Fallback: just pick any valid antichain?
             # But min_terms was valid.
             raise ValueError(f"Reconstruction failed for {target}")
             
        results[target] = path

    return results[1:]
