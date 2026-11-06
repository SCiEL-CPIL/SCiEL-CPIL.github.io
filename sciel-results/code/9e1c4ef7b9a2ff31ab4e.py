def construct():
    from fractions import Fraction
    import sys
    
    # Increase recursion limit just in case, though depth is max 47
    sys.setrecursionlimit(10000)

    denominators = list(range(2, 49))
    n = len(denominators)
    values = [Fraction(1, d) for d in denominators]
    
    # Sort by value descending (denominator ascending)
    # indices in original order: 0 is 1/2, 1 is 1/3, etc.
    # Since denominators are 2..48, value 1/d is decreasing as d increases.
    # So sorting by value descending is same as sorting by denominator ascending.
    # The list is already in that order.
    
    # Precompute suffix sums of values
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + values[i]
        
    best_sum = Fraction(0)
    best_indices = []
    
    # To speed up, we can initialize with a known good solution or just let it find one.
    # The DFS will find the optimal one.
    
    def dfs(i, current_sum, current_indices):
        nonlocal best_sum, best_indices
        
        # Pruning
        if current_sum >= 1:
            return
            
        # If even taking all remaining doesn't beat best_sum, prune
        if current_sum + suffix_sums[i] <= best_sum:
            return
            
        # If taking all remaining keeps us below 1, that's the best we can do from here
        if current_sum + suffix_sums[i] < 1:
            candidate_sum = current_sum + suffix_sums[i]
            if candidate_sum > best_sum:
                best_sum = candidate_sum
                # Indices are current_indices + all indices from i to n-1
                best_indices = current_indices + list(range(i, n))
            return
            
        if i == n:
            if 0 < current_sum < 1 and current_sum > best_sum:
                best_sum = current_sum
                best_indices = current_indices[:]
            return

        # Option 1: Include item i
        # Check if including it makes sum >= 1? 
        # The recursion handles that, but we can prune early if current_sum + values[i] >= 1 
        # and we know we can't add more? No, we might exclude later ones. 
        # But if current_sum + values[i] >= 1, we cannot include i AND any further items that push it over? 
        # Actually, if current_sum + values[i] >= 1, we can still include i if we exclude all subsequent? 
        # No, if current_sum + values[i] >= 1, then the sum is already >= 1. 
        # Wait, the condition is STRICTLY BELOW 1. 
        # So if current_sum + values[i] >= 1, we cannot take this branch? 
        # Not necessarily. If we take i, sum is >= 1. Invalid. 
        # So we can prune the "Include" branch if current_sum + values[i] >= 1.
        
        if current_sum + values[i] < 1:
            current_indices.append(i)
            dfs(i + 1, current_sum + values[i], current_indices)
            current_indices.pop()
            
        # Option 2: Exclude item i
        dfs(i + 1, current_sum, current_indices)
        
    dfs(0, Fraction(0), [])
    
    return sorted(best_indices)
