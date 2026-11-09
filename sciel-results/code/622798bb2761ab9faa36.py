def construct():
    import time
    import math
    import sys

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
        
    # Greedy Initialization
    best_sum = 0
    best_mask = 0
    current_sum = 0
    current_mask = 0
    for i in range(n):
        if current_sum + weights[i] < L:
            current_sum += weights[i]
            current_mask |= (1 << i)
    if current_sum > best_sum:
        best_sum = current_sum
        best_mask = current_mask

    # Iterative DFS
    # Stack entries: (idx, current_sum, mask)
    # Push Exclude then Include to pop Include first
    stack = [(0, 0, 0)]
    
    while stack:
        if time.time() > deadline:
            break
            
        idx, cur_sum, mask = stack.pop()
        
        # Pruning: Can we beat best_sum?
        if cur_sum + suffix_sums[idx] <= best_sum:
            continue
            
        # Update best
        if cur_sum > best_sum:
            best_sum = cur_sum
            best_mask = mask
            
        if idx == n:
            continue
            
        w = weights[idx]
        
        # Option 2: Exclude (push first)
        stack.append((idx + 1, cur_sum, mask))
        
        # Option 1: Include (push second)
        new_sum = cur_sum + w
        if new_sum < L:
            stack.append((idx + 1, new_sum, mask | (1 << idx)))

    # Reconstruct indices from best_mask
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(indices[i])
            
    # If no result found (should not happen as greedy finds at least one), return [0]
    if not result:
        result = [0]
        
    return result
