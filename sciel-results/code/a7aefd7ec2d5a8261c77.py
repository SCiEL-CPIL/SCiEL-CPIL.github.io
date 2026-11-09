"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    # Known optimal candidate from previous observations
    best_solution = [8, 31, 248]
    best_max_denom = 248
    
    target = Fraction(5, 31)
    # We search for solutions with max denominator < 248
    # If found, we will have a score > 7.0
    # If not found, we return the known best.
    
    # Precompute suffix sums of reciprocals for pruning
    # suffix_sum[i] = sum(1/k for k in range(i, 248))
    # We use Fractions for exact comparison
    max_search_limit = 247
    suffix_sum = [Fraction(0, 1)] * (max_search_limit + 2)
    s = Fraction(0, 1)
    for i in range(max_search_limit, 0, -1):
        s += Fraction(1, i)
        suffix_sum[i] = s
    
    def dfs(rem, last_d, path):
        nonlocal best_solution, best_max_denom
        
        if rem == 0:
            # Found a valid solution
            if len(path) <= 256:
                current_max = path[-1]
                if current_max < best_max_denom:
                    best_max_denom = current_max
                    best_solution = path[:]
            return
        
        if rem < 0:
            return
            
        # The next denominator d must satisfy:
        # 1. d > last_d (distinctness and increasing order)
        # 2. 1/d <= rem  => d >= ceil(1/rem)
        # 3. d < best_max_denom (to improve current best)
        
        min_d_from_rem = (rem.denominator + rem.numerator - 1) // rem.numerator
        min_d = max(last_d + 1, min_d_from_rem)
        
        # If the smallest possible next denominator is already >= best_max_denom, 
        # we cannot find a better solution in this branch.
        if min_d >= best_max_denom:
            return
            
        # Also, we cannot exceed max_search_limit (which is best_max_denom - 1 initially)
        # But best_max_denom might decrease during search.
        # We iterate d from min_d to best_max_denom - 1
        
        for d in range(min_d, best_max_denom):
            # Pruning: Can we complete the sum with denominators > d?
            # The remaining sum after picking d is rem - 1/d.
            # The max sum we can add with denominators d+1 to best_max_denom-1 is suffix_sum[d+1].
            # Note: suffix_sum is computed up to 247. If best_max_denom decreases, 
            # the actual available range is smaller, so suffix_sum[d+1] is an upper bound 
            # on the possible sum. If even this upper bound is < remaining, we prune.
            
            next_rem = rem - Fraction(1, d)
            
            if next_rem < 0:
                # Since d is increasing, 1/d is decreasing, so next_rem will increase.
                # Wait, if next_rem < 0, it means 1/d > rem.
                # But we ensured d >= ceil(1/rem), so 1/d <= rem.
                # So next_rem should be >= 0.
                # However, due to integer division in ceil, let's be safe.
                # Actually, if d >= ceil(1/rem), then d >= 1/rem, so 1/d <= rem.
                # So next_rem >= 0.
                pass
            
            # Check if max possible sum from d+1 onwards is sufficient
            # We need to sum to next_rem using denominators > d.
            # The max sum is suffix_sum[d+1].
            if suffix_sum[d+1] < next_rem:
                continue
            
            path.append(d)
            dfs(next_rem, d, path)
            path.pop()
            
            # If we found a better solution, best_max_denom has been updated.
            # The loop condition `d < best_max_denom` will automatically shrink the range
            # for subsequent iterations in this loop if best_max_denom decreases.
            # However, Python's range object is static. 
            # We should check if we should break.
            # If best_max_denom has decreased to <= d, we can break.
            if d >= best_max_denom:
                break

    # Start search
    dfs(target, 0, [])
    
    return best_solution
# EVOLVE-BLOCK-END
