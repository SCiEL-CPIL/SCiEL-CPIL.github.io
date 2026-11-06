from fractions import Fraction

def construct():
    # The minimum number of distinct unit fractions with denominators >= 5
    # summing to 1 is 8.
    # We search for k from 5 upwards.
    # For k=5,6,7, the maximum possible sum (using smallest denominators) is < 1.
    # Max sum for k=7: 1/5+1/6+1/7+1/8+1/9+1/10+1/11 = 0.932 < 1.
    # So k must be at least 8.
    
    def find_solution(k):
        # Use backtracking with tight bounds
        # State: idx, current_sum (Fraction), last_denom
        # Target: 1
        
        def dfs(idx, current_sum, last_denom):
            if idx == k:
                return current_sum == 1
            
            remaining_terms = k - idx
            rem_sum = 1 - current_sum
            
            # Pruning: if rem_sum <= 0, invalid
            if rem_sum <= 0:
                return None
            
            # Determine bounds for next denominator d
            # 1. d > last_denom
            # 2. d >= 5
            # 3. 1/d <= rem_sum  => d >= 1/rem_sum
            # 4. Sum of remaining_terms terms starting from d must be >= rem_sum
            #    Max sum is approx remaining_terms / d (actually sum 1/(d+i))
            #    A tighter lower bound on d is not direct, but an upper bound on d is:
            #    If d is too large, 1/d is too small, and even the smallest possible
            #    subsequent denominators (d+1, d+2...) might not sum to rem_sum?
            #    Actually, if d is large, the sum is small.
            #    We need sum_{i=0}^{remaining_terms-1} 1/(d+i) >= rem_sum.
            #    This gives an upper bound on d.
            #    Approx: remaining_terms / d >= rem_sum => d <= remaining_terms / rem_sum.
            
            min_d = max(last_denom + 1, 5)
            # d must be at least ceil(1/rem_sum)
            # 1/d <= rem_sum => d >= 1/rem_sum
            min_d = max(min_d, int(1 / rem_sum) + 1) # Careful with integer division, use Fraction logic
            # Actually: 1/d <= rem_sum => d >= 1/rem_sum.
            # If rem_sum is Fraction, 1/rem_sum is Fraction.
            # d is integer. d >= ceil(1/rem_sum).
            
            # Let's compute bounds precisely
            # Lower bound for d:
            # d >= last_denom + 1
            # d >= 5
            # 1/d <= rem_sum => d >= 1/rem_sum
            lb1 = last_denom + 1
            lb2 = 5
            lb3 = int(1 / rem_sum) + 1 # Ceiling of 1/rem_sum
            
            start_d = max(lb1, lb2, lb3)
            
            # Upper bound for d:
            # We need to be able to reach rem_sum with remaining_terms terms.
            # The maximum sum we can get with remaining_terms terms starting from d is:
            # S_max(d) = sum_{i=0}^{remaining_terms-1} 1/(d+i)
            # We need S_max(d) >= rem_sum.
            # Since S_max(d) is decreasing with d, we can find the max d.
            # Approx: remaining_terms / d >= rem_sum => d <= remaining_terms / rem_sum
            ub = int(remaining_terms / rem_sum)
            
            # Also, we must ensure d is not too small such that 1/d > rem_sum
            # This is handled by start_d >= ceil(1/rem_sum)
            
            # Iterate d from start_d to ub
            for d in range(start_d, ub + 1):
                # Check if 1/d <= rem_sum
                if Fraction(1, d) > rem_sum:
                    continue
                
                new_sum = current_sum + Fraction(1, d)
                
                # Recursive call
                res = dfs(idx + 1, new_sum, d)
                if res:
                    return [d] + res
            
            return None
        
        return dfs(0, Fraction(0), 4)

    for k in range(5, 15):
        sol = find_solution(k)
        if sol:
            return sol
    
    # Fallback: Greedy (should not be reached for k>=8)
    rem = Fraction(1)
    denoms = []
    last = 4
    while rem > 0:
        d = last + 1
        while Fraction(1, d) > rem:
            d += 1
        denoms.append(d)
        rem -= Fraction(1, d)
        last = d
        if len(denoms) > 256:
            break
    return denoms
