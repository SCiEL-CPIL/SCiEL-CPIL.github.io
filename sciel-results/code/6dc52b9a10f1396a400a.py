from fractions import Fraction

def construct():
    # Target sum is 1
    target = Fraction(1, 1)
    min_denom = 5
    max_denom = 10000000
    
    # Try k from 8 upwards (theoretical min is 8)
    for k in range(8, 257):
        result = _search(k, min_denom, target)
        if result:
            return result
            
    raise RuntimeError("No solution found")

def _search(terms_left, start_denom, remaining_sum):
    if terms_left == 0:
        if remaining_sum == 0:
            return []
        return None
    
    if terms_left == 1:
        # Check if remaining_sum is a unit fraction with denom >= start_denom
        if remaining_sum.numerator == 1:
            d = remaining_sum.denominator
            if d >= start_denom and d <= 10000000:
                return [d]
        return None

    # Lower bound for d: 1/d <= remaining_sum => d >= 1/remaining_sum
    # Also d >= start_denom
    min_d = max(start_denom, (1 // remaining_sum.numerator) * remaining_sum.denominator) # ceil(1/remaining_sum)
    # Actually, ceil(1/x) for x = p/q is ceil(q/p).
    # q // p + (1 if q % p else 0)
    
    # Let's calculate min_d carefully
    # 1/d <= p/q => q <= p*d => d >= q/p
    p, q = remaining_sum.numerator, remaining_sum.denominator
    min_d = max(start_denom, (q + p - 1) // p)
    
    # Upper bound for d:
    # We need to be able to form remaining_sum - 1/d with (terms_left - 1) terms.
    # The maximum sum of (terms_left - 1) terms starting from d+1 is M.
    # remaining_sum - 1/d <= M
    # Also, the smallest possible sum is > 0.
    # We iterate d from min_d upwards.
    
    # We can also estimate a max_d.
    # If 1/d is too small, the remaining sum is too large to be covered by the remaining terms?
    # No, if 1/d is small, remaining_sum - 1/d is large.
    # The max sum of remaining terms decreases as d increases.
    # So there is a maximum d for which a solution exists.
    
    for d in range(min_d, 10000000):
        # Pruning: Check if remaining sum can be formed
        next_sum = remaining_sum - Fraction(1, d)
        if next_sum < 0:
            break # d is too small? No, d is increasing, 1/d decreasing, next_sum increasing.
            # Wait. If d is small, 1/d is large. next_sum = R - 1/d.
            # If 1/d > R, next_sum < 0. This happens for small d.
            # But we started d from min_d where 1/d <= R. So next_sum >= 0.
            # As d increases, 1/d decreases, so next_sum increases.
            # So next_sum will never be < 0 in this loop.
            
        # Max possible sum for remaining terms
        # Sum of 1/(d+1) + ... + 1/(d + terms_left - 1)
        # Approximation for pruning:
        # If terms_left - 1 is large, this sum is approx (terms_left - 1) / (d + 1)
        # Let's compute it exactly if terms_left is small, or approximate.
        # For k=8, terms_left is small.
        
        max_next_sum = Fraction(0, 1)
        for j in range(1, terms_left):
            max_next_sum += Fraction(1, d + j)
            
        if next_sum > max_next_sum:
            # It is impossible to form next_sum with terms starting from d+1
            # Since increasing d will decrease max_next_sum and increase next_sum (slightly), 
            # the condition will only get worse.
            break
            
        # Recurse
        res = _search(terms_left - 1, d + 1, next_sum)
        if res:
            return [d] + res
            
    return None
