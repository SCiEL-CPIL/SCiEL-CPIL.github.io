import time

def construct():
    """
    Finds the subset of denominators [2, 3, ..., 48] that minimizes the deficit 
    1 - sum(1/d) subject to sum(1/d) < 1.
    
    Uses a branch-and-bound approach with floating-point arithmetic for speed.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute reciprocal values as floats
    reciprocals = [1.0 / d for d in denominators]
    
    # Precompute suffix sums for pruning
    # suffix_sums[i] is the sum of reciprocals[i:]
    suffix_sums = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + reciprocals[i]
        
    # Global best state
    best_sum = -1.0
    best_indices = []
    
    # We want to find the subset with sum closest to 1 from below.
    # DFS traversal
    def dfs(i, cur_sum, indices):
        nonlocal best_sum, best_indices
        
        # Time check
        if time.time() > deadline:
            return
            
        # Pruning 1: If current sum is already >= 1, we can't use this path
        # because adding more positive terms will only increase it.
        if cur_sum >= 1.0:
            return
            
        # Pruning 2: Even if we add all remaining items, can we beat the best?
        # Max possible sum from this state is cur_sum + suffix_sums[i]
        if cur_sum + suffix_sums[i] <= best_sum:
            return
            
        # If we've considered all items, check if this is the best
        if i == n:
            if cur_sum > best_sum:
                best_sum = cur_sum
                best_indices = indices[:]
            return
            
        # Option 1: Exclude item i
        dfs(i + 1, cur_sum, indices)
        
        # Option 2: Include item i
        new_sum = cur_sum + reciprocals[i]
        # We don't strictly need to check new_sum < 1 here because the top of the 
        # recursive call will handle it, but we can prune early if we know it will fail.
        # However, if new_sum < 1, we proceed. If new_sum >= 1, the next call returns immediately.
        # To save a function call, we can check here:
        if new_sum < 1.0:
             dfs(i + 1, new_sum, indices + [i])
        else:
             # If new_sum >= 1, we can't include this item in any valid solution 
             # that extends this state, so we just skip the recursive call.
             pass

    # Start DFS
    dfs(0, 0.0, [])
    
    # Fallback if no solution found (should not happen as [0] gives 1/2 < 1)
    if not best_indices:
        return [0]
        
    return best_indices

if __name__ == "__main__":
    res = construct()
    # print(res)
    pass
