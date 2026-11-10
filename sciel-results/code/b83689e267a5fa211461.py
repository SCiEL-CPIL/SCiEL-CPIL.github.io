"""Improved program: DFS search for max denominator < 248, fallback to [8, 31, 248]."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    
    target = Fraction(5, 31)
    best_solution = [8, 31, 248]
    best_max = 248
    
    # We try to find a solution with max denominator < 248.
    # If found, we update best_solution and best_max.
    # We limit the search depth to 10 terms to keep it fast.
    # We limit the max denominator to 247.
    
    def dfs(remaining, min_d, current_denoms, depth):
        nonlocal best_solution, best_max
        if remaining == 0:
            if depth > 0:
                # Found a valid solution
                max_d = max(current_denoms)
                if max_d < best_max:
                    best_max = max_d
                    best_solution = current_denoms[:]
            return
        
        if depth == 0:
            return
            
        # Bounds for d
        # 1/d < remaining => d > 1/remaining
        # Sum of k smallest distinct reciprocals starting at d must be <= remaining?
        # No, we need to hit 'remaining' exactly.
        # The maximum sum we can form with 'depth' terms starting at min_d is roughly depth/min_d.
        # If depth/min_d < remaining, we can't reach remaining with these small denominators?
        # Wait, if min_d is small, 1/min_d is large.
        # We need to pick d such that 1/d <= remaining.
        # Also, the sum of the 'depth' largest possible reciprocals (smallest denominators)
        # starting from min_d must be >= remaining.
        # Actually, if we pick d, the next min is d+1.
        # Max possible sum with 'depth' terms starting at d is 1/d + 1/(d+1) + ...
        # If this max sum < remaining, then d is too large (denominator too big, reciprocals too small).
        # So we need 1/d + 1/(d+1) + ... + 1/(d+depth-1) >= remaining.
        
        # Lower bound for d:
        # 1/d < remaining is not strictly true, 1/d can be equal? No, distinct denominators, but 1/d is a single term.
        # If 1/d > remaining, we can't use it. So 1/d <= remaining => d >= 1/remaining.
        # Actually, if 1/d > remaining, we skip.
        
        # Upper bound for d:
        # We need to be able to sum to 'remaining' with 'depth' terms.
        # The smallest sum we can make with 'depth' terms starting at d is 1/d + 1/(d+1) + ...
        # If this smallest sum > remaining, then d is too small (denominator too small, reciprocals too big).
        # Wait, if d is small, 1/d is big.
        # So if 1/d > remaining, we skip (d too small).
        # If 1/d + 1/(d+1) + ... + 1/(d+depth-1) < remaining, then even the smallest denominators (largest reciprocals)
        # starting from d are not enough? No, if d is small, reciprocals are large.
        # Let's check:
        # If d=10, depth=2, remaining=0.2.
        # 1/10 + 1/11 = 0.1 + 0.09 = 0.19 < 0.2.
        # So we can't reach 0.2 with denominators >= 10 and 2 terms?
        # We could use 1/10 + 1/9? No, distinct and increasing?
        # The problem doesn't require increasing order in the list, but we can assume w.l.o.g. $d_1 < d_2 < \dots$ for search.
        # So if we enforce $d_i < d_{i+1}$, then the next denominators are $> d$.
        # So the max sum of remaining 'depth' terms given current min_d is $1/min\_d + 1/(min\_d+1) + \dots$
        # If this max sum < remaining, then we can't reach remaining. We need smaller denominators (smaller d).
        # But we are iterating d upwards from min_d.
        # So if at d=min_d, the max sum < remaining, we should return? No, we start at min_d.
        # If the max sum starting at min_d is < remaining, then NO solution exists for this branch.
        
        # Let's calculate max_sum for given d and depth.
        # max_sum = sum(1/(d+i) for i in range(depth))
        
        # We iterate d from min_d to 247.
        # Condition 1: 1/d <= remaining (if 1/d > remaining, d is too small? No, if 1/d > remaining, we can't use d as the next term because the sum would exceed remaining? 
        # Wait, if 1/d > remaining, then adding 1/d makes sum > target. So we must have 1/d <= remaining.
        # This implies d >= 1/remaining.
        
        # Condition 2: The sum of the 'depth' largest possible reciprocals starting at d must be >= remaining.
        # i.e., 1/d + 1/(d+1) + ... + 1/(d+depth-1) >= remaining.
        # If this is false, then even with the smallest possible denominators (largest reciprocals), we can't reach 'remaining'.
        # So d is too large? No, if d is larger, reciprocals are smaller, sum is smaller.
        # So if sum < remaining, we need smaller d.
        # But we are iterating d upwards. So if it fails at min_d, it fails for all larger d.
        # So we can break/return.
        
        # Let's refine:
        # We want to find d such that:
        # 1. 1/d <= remaining
        # 2. remaining - 1/d can be represented by depth-1 terms > d.
        
        # To bound d:
        # Max possible sum of depth terms starting at d is S_max(d) = sum(1/(d+i) for i in range(depth)).
        # If S_max(d) < remaining, then d is too large (reciprocals too small).
        # Since S_max(d) decreases as d increases, if S_max(min_d) < remaining, we return.
        # We iterate d from min_d.
        # We stop when 1/d > remaining? No, 1/d <= remaining is required.
        # So d >= ceil(1/remaining).
        
        start_d = max(min_d, (1 / remaining).numerator // (1 / remaining).denominator + 1) # 1/remaining
        # Actually, if 1/d > remaining, we skip.
        # So d must be such that 1/d <= remaining.
        # d >= 1/remaining.
        
        # Let's just iterate and check.
        
        # Optimization: If depth == 1:
        if depth == 1:
            # Need 1/d = remaining => d = 1/remaining
            if remaining.denominator % remaining.numerator == 0:
                d = remaining.denominator // remaining.numerator
                if d >= min_d and d <= 247:
                    current_denoms.append(d)
                    dfs(0, d+1, current_denoms, 0)
                    current_denoms.pop()
            return

        # Iterate d
        # Upper bound for d:
        # We need S_max(d) >= remaining.
        # S_max(d) approx depth/d.
        # So depth/d >= remaining => d <= depth/remaining.
        end_d = min(247, int(depth / remaining) + 1)
        
        for d in range(start_d, end_d + 1):
            # Check if 1/d <= remaining
            if Fraction(1, d) > remaining:
                break # d is too small? No, if 1/d > remaining, d is small.
                # Wait, if d increases, 1/d decreases.
                # So if 1/d > remaining, we are at small d. We need larger d.
                # So we should continue?
                # If 1/d > remaining, we can't use d.
                # But larger d will have smaller 1/d.
                # So we continue until 1/d <= remaining.
                continue
            
            # Check if remaining - 1/d is representable by depth-1 terms > d.
            # Max sum of depth-1 terms > d is 1/(d+1) + ... + 1/(d+depth-1).
            # If this max sum < remaining - 1/d, then we can't reach it.
            # This condition is equivalent to S_max(d) < remaining.
            # We already bounded d such that S_max(d) >= remaining (approx).
            # Let's check exact sum for small depth.
            
            rem2 = remaining - Fraction(1, d)
            
            # Pruning:
            # Max sum of depth-1 terms starting at d+1:
            max_rem_sum = sum(Fraction(1, d+i) for i in range(1, depth))
            if max_rem_sum < rem2:
                # d is too large?
                # If d increases, max_rem_sum decreases.
                # So if it's too small here, it will be too small for larger d.
                # So we can break.
                break
                
            # Min sum of depth-1 terms starting at d+1?
            # The denominators must be distinct and > d.
            # The smallest sum is 1/(d+1) + 1/(d+2) + ... ? No, these are the LARGEST reciprocals.
            # The smallest sum would be with the LARGEST denominators.
            # But we don't have an upper bound on the next denominators other than 247.
            # However, if rem2 is very small, we might need large denominators.
            # But we are capped at 247.
            # If the minimum possible sum (using largest available denominators) > rem2, then we can't reach rem2.
            # Min sum of depth-1 terms from {d+1, ..., 247}:
            # Take the largest depth-1 denominators: 247, 246, ...
            # If depth-1 > 247 - d, we can't pick distinct denominators.
            if 247 - d < depth - 1:
                continue # Not enough denominators left
                
            # This check is expensive. Let's just recurse.
            current_denoms.append(d)
            dfs(rem2, d+1, current_denoms, depth-1)
            current_denoms.pop()
            
    # Start search
    # We try to find a solution with max < 248.
    # We allow up to 10 terms.
    # We start with min_d = 1.
    # But 1/1 = 1 > 5/31. So start_d will be adjusted.
    
    dfs(target, 1, [], 10)
    
    return best_solution
# EVOLVE-BLOCK-END
