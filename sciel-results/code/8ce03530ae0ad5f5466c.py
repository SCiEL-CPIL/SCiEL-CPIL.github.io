import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s deadline to be safe
    
    # Numbers 2 to 256
    nums = list(range(2, 257))
    n = len(nums)
    
    # Sort by weight (1/x) descending, which is x ascending
    # We want to consider small numbers first as they have high weight
    # But wait, standard B&B for Max Weight Independent Set usually sorts by weight descending.
    # Here weight is 1/x. So sorting x ascending is correct.
    # However, the upper bound is the sum of remaining weights.
    # If we sort by weight descending, the "suffix sum" of weights is the best possible upper bound.
    
    # Let's stick to x ascending (2, 3, ..., 256)
    # weights[i] = 1.0 / nums[i]
    
    # Precompute conflict masks
    # conflict_mask[i] is a bitmask where bit j is 1 if nums[i] and nums[j] conflict
    # Conflict means lcm(nums[i], nums[j]) <= 256
    conflict_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if lcm(nums[i], nums[j]) <= 256:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
                
    # Suffix sum of weights for upper bound
    # suffix_sum[k] = sum of weights from k to n-1
    suffix_sum = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + (1.0 / nums[i])
        
    best_sum = 0.0
    best_set = []
    
    def solve(k, selected_mask, current_sum, selected_indices):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
            
        if k == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = list(selected_indices)
            return
            
        # Upper bound: current_sum + sum of all remaining weights
        if current_sum + suffix_sum[k] <= best_sum:
            return
            
        # Option 1: Include nums[k]
        # Check if nums[k] conflicts with any selected node
        if (selected_mask & conflict_mask[k]) == 0:
            selected_indices.append(k)
            solve(k + 1, selected_mask | (1 << k), current_sum + 1.0 / nums[k], selected_indices)
            selected_indices.pop()
            
        # Option 2: Exclude nums[k]
        # Heuristic: If including nums[k] was possible and improved the sum, we might have already found a good solution.
        # But we must explore exclusion too.
        # To improve pruning, we can check if excluding nums[k] can possibly beat best_sum.
        # The upper bound for exclusion is current_sum + suffix_sum[k+1].
        if current_sum + suffix_sum[k + 1] > best_sum:
            solve(k + 1, selected_mask, current_sum, selected_indices)
            
    solve(0, 0, 0.0, [])
    
    return [nums[i] for i in best_set]
