# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import time
    
    deadline = time.time() + 80
    
    denoms = list(range(2, 49))
    values = [Fraction(1, d) for d in denoms]
    n = len(values)
    
    # Precompute suffix sums
    suffix_sum = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + values[i]
    
    # Known good solution: 1/2 + 1/3 + 1/7 + 1/43 = 1805/1806
    # Indices: 0, 1, 5, 41 (denom 43 is index 41)
    initial_indices = [0, 1, 5, 41]
    initial_sum = sum(values[i] for i in initial_indices)
    
    best_sum = initial_sum
    best_indices = list(initial_indices)
    
    def branch_and_bound(idx, current_sum, current_indices):
        nonlocal best_sum, best_indices
        if time.time() > deadline:
            return
        
        if idx == n:
            if current_sum < 1 and current_sum > best_sum:
                best_sum = current_sum
                best_indices = list(current_indices)
            return
        
        # Upper bound
        upper = current_sum + suffix_sum[idx]
        if upper <= best_sum:
            return
        
        # Try including first
        if current_sum + values[idx] < 1:
            current_indices.append(idx)
            branch_and_bound(idx + 1, current_sum + values[idx], current_indices)
            current_indices.pop()
            
            if time.time() > deadline:
                return
        
        # Try excluding
        branch_and_bound(idx + 1, current_sum, current_indices)
    
    branch_and_bound(0, Fraction(0), [])
    
    return best_indices
# EVOLVE-BLOCK-END
