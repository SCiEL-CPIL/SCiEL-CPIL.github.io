def construct():
    import time
    from fractions import Fraction
    
    start_time = time.time()
    DEADLINE = 85.0  # 5s safety margin
    
    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    
    # Sort by descending reciprocal value = ascending denominator (already sorted)
    # So denominators[0]=2 has largest reciprocal 1/2
    
    # Precompute suffix sums as floats for pruning
    suffix_sum = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + 1.0 / denominators[i]
    
    best_sum = 0.0
    best_indices = []
    
    # DFS
    def dfs(i, current_sum, selected):
        nonlocal best_sum, best_indices
        
        if time.time() > DEADLINE:
            return
        
        # If we've considered all items
        if i == n:
            if 0 < current_sum < 1.0 and current_sum > best_sum:
                best_sum = current_sum
                best_indices = list(selected)
            return
        
        # Pruning: even if we take all remaining, can we beat best?
        # Max possible sum from here: current_sum + suffix_sum[i]
        # But we're constrained by < 1, so effective max is min(current_sum + suffix_sum[i], 1.0)
        # If current_sum + suffix_sum[i] <= best_sum, prune
        if current_sum + suffix_sum[i] <= best_sum:
            return
        
        # Option 1: Skip item i
        dfs(i + 1, current_sum, selected)
        
        # Option 2: Take item i (if it keeps sum < 1)
        new_sum = current_sum + 1.0 / denominators[i]
        if new_sum < 1.0:
            selected.append(i)
            dfs(i + 1, new_sum, selected)
            selected.pop()
    
    dfs(0, 0.0, [])
    
    # Verify with exact arithmetic
    if best_indices:
        exact_sum = sum(Fraction(1, denominators[i]) for i in best_indices)
        if exact_sum < 1:
            return best_indices
    else:
        # Fallback: greedy
        selected, total = [], Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected
    
    return best_indices
