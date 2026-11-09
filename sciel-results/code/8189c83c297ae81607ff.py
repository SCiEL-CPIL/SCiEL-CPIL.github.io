def construct():
    from fractions import Fraction
    
    denominators = list(range(2, 49))
    n = len(denominators)
    reciprocals = [Fraction(1, d) for d in denominators]
    
    # Sort indices by reciprocal value descending (denominator ascending)
    # We need to return original indices
    order = sorted(range(n), key=lambda i: -reciprocals[i])
    
    # Precompute suffix sums of reciprocals in the sorted order
    sorted_recips = [reciprocals[i] for i in order]
    suffix_sum = [Fraction() for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + sorted_recips[i]
    
    best_sum = Fraction()
    best_indices = []
    
    def dfs(i, current_sum, current_indices):
        nonlocal best_sum, best_indices
        
        # Prune: if current sum >= 1, stop
        if current_sum >= 1:
            return
            
        # Prune: if current sum + max possible remaining sum <= best_sum, stop
        if current_sum + suffix_sum[i] <= best_sum:
            return
            
        # Update best if we have a valid sum
        if current_sum > best_sum:
            best_sum = current_sum
            best_indices = list(current_indices)
            
        # If we've considered all elements, stop
        if i == n:
            return
            
        # Option 1: Include current element
        if current_sum + sorted_recips[i] < 1:
            current_indices.append(order[i])
            dfs(i + 1, current_sum + sorted_recips[i], current_indices)
            current_indices.pop()
            
        # Option 2: Exclude current element
        dfs(i + 1, current_sum, current_indices)
        
    dfs(0, Fraction(), [])
    
    return best_indices
