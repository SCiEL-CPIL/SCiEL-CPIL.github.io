"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # items: (denominator, original_index, reciprocal)
    items = [(d, i, Fraction(1, d)) for i, d in enumerate(denominators)]
    # Sort by denominator descending
    items.sort(key=lambda x: x[0], reverse=True)
    
    # Precompute suffix sums of reciprocals
    suffix_sum = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + items[i][2]
        
    best_sum = Fraction(0)
    best_indices = []
    
    def dfs(idx, current_sum, current_indices):
        nonlocal best_sum, best_indices
        
        if current_sum >= 1:
            return
            
        if current_sum + suffix_sum[idx] <= best_sum:
            return
            
        if current_sum > best_sum:
            best_sum = current_sum
            best_indices = current_indices[:]
            
        if idx == n:
            return
            
        # Include
        next_sum = current_sum + items[idx][2]
        if next_sum < 1:
            current_indices.append(items[idx][1])
            dfs(idx + 1, next_sum, current_indices)
            current_indices.pop()
            
        # Exclude
        dfs(idx + 1, current_sum, current_indices)

    dfs(0, Fraction(0), [])
    
    return best_indices
# EVOLVE-BLOCK-END
