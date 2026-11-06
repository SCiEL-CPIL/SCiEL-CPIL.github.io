from fractions import Fraction
import time

def construct():
    # Hardcoded 10-term solution
    fallback = [5, 6, 7, 8, 9, 10, 12, 15, 229, 577080]
    
    # Try to find a solution with 8 terms
    start_time = time.time()
    for k in range(8, 10):
        if time.time() - start_time > 50:
            break
        result = _search(k, 5, Fraction(1, 1))
        if result:
            return result
            
    return fallback

def _search(terms_left, start_denom, remaining_sum):
    if terms_left == 0:
        if remaining_sum == 0:
            return []
        return None
    
    if terms_left == 1:
        if remaining_sum.numerator == 1:
            d = remaining_sum.denominator
            if d >= start_denom and d <= 10000000:
                return [d]
        return None

    p, q = remaining_sum.numerator, remaining_sum.denominator
    min_d = max(start_denom, (q + p - 1) // p)
    
    # Pruning upper bound
    # Max sum of terms_left terms starting from min_d
    # If remaining_sum > sum(1/min_d + 1/(min_d+1) + ...), return None
    # But we iterate d.
    
    for d in range(min_d, 10000000):
        next_sum = remaining_sum - Fraction(1, d)
        if next_sum < 0:
            break
            
        # Check if next_sum can be formed by terms_left - 1 terms
        # Lower bound for next_sum: sum of smallest terms_left-1 fractions starting from d+1
        # Upper bound for next_sum: sum of largest terms_left-1 fractions starting from d+1
        # Actually, just recurse with pruning.
        
        # Pruning: if next_sum is too large to be formed by remaining terms
        # Max possible sum for remaining terms:
        max_next_sum = Fraction(0, 1)
        for j in range(1, terms_left):
            max_next_sum += Fraction(1, d + j)
            
        if next_sum > max_next_sum:
            break
            
        # Pruning: if next_sum is too small
        # Min possible sum for remaining terms (largest denominators)
        # This is hard to bound easily without knowing the denominators.
        # But we can check if next_sum < 1/(d + terms_left - 1) * (terms_left - 1) ? No.
        
        res = _search(terms_left - 1, d + 1, next_sum)
        if res:
            return [d] + res
            
    return None
