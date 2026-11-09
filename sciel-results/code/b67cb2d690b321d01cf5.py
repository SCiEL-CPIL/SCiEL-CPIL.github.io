def construct():
    import time
    import math
    import sys

    # Increase recursion limit for DFS
    sys.setrecursionlimit(10000)

    start_time = time.time()
    deadline = start_time + 55  # Safety margin

    # Denominators 2 to 48
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of 2 to 48
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Weights are L / d
    # Sort by weight descending (i.e., denominator ascending)
    # But we need to keep track of original indices
    items = []
    for i, d in enumerate(denominators):
        w = L // d
        items.append((w, i))
        
    # Sort by weight descending
    items.sort(key=lambda x: x[0], reverse=True)
    
    # Extract weights and indices
    weights = [x[0] for x in items]
    indices = [x[1] for x in items]
    
    # Suffix sums for pruning
    suffix_sums = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + weights[i]
        
    best_sum = 0
    best_indices_set = set()
    
    # B&B DFS
    # idx: current index in sorted items
    # current_sum: sum of weights selected so far
    # selected_indices: list of original indices selected so far
    def dfs(idx, current_sum, selected_indices):
        nonlocal best_sum, best_indices_set, deadline
        
        if time.time() > deadline:
            return
        
        # Pruning 1: Even if we take all remaining, we can't beat best_sum
        if current_sum + suffix_sums[idx] <= best_sum:
            return
            
        # Update best if current sum is valid and better
        if current_sum > best_sum:
            best_sum = current_sum
            best_indices_set = set(selected_indices)
            
        if idx == n:
            return
            
        # Option 1: Include current item
        new_sum = current_sum + weights[idx]
        if new_sum < L: # Must be strictly below 1
            # Recurse
            selected_indices.append(indices[idx])
            dfs(idx + 1, new_sum, selected_indices)
            selected_indices.pop()
        # If new_sum >= L, we cannot include it (and any larger sum would also fail, but we are adding positive weights)
        # Wait, if new_sum >= L, we just skip the include branch. We don't prune the exclude branch.
        
        # Option 2: Exclude current item
        dfs(idx + 1, current_sum, selected_indices)

    dfs(0, 0, [])
    
    # Convert set to list of indices
    result = list(best_indices_set)
    
    # If no result found (should not happen as {0} is valid), return [0]
    if not result:
        result = [0]
        
    return result
