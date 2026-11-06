import time
from fractions import Fraction

def construct():
    deadline = time.time() + 85.0
    
    def max_sum_from(start, count):
        """Maximum sum of 'count' unit fractions with denominators starting from 'start'."""
        if count <= 0:
            return Fraction(0, 1)
        if count == 1:
            return Fraction(1, start)
        # For small counts, compute directly
        s = Fraction(0, 1)
        for i in range(count):
            s += Fraction(1, start + i)
        return s
    
    def search(target, min_denom, terms_left, current):
        if time.time() > deadline:
            return None
        
        if terms_left == 0:
            if target == 0:
                return current
            return None
        
        if terms_left == 1:
            if target.numerator == 1:
                d = target.denominator
                if d >= min_denom and d >= 5 and d <= 10000000 and d not in current:
                    return current + [d]
            return None
        
        if target <= 0:
            return None
        
        # Max possible sum with terms_left terms from min_denom
        max_possible = max_sum_from(min_denom, terms_left)
        if target > max_possible:
            return None
        
        # d > 1/target
        d_min_from_target = target.denominator // target.numerator + 1
        d_start = max(min_denom, 5, d_min_from_target)
        
        # Upper bound: target - 1/d must be <= max_sum_from(d+1, terms_left-1)
        # So 1/d >= target - max_sum_from(d+1, terms_left-1)
        # This is hard to solve for d directly, so we iterate
        
        # Simple upper bound: 1/d >= target - (terms_left-1)/5
        # If target - (terms_left-1)/5 > 0, then d <= 1/(target - (terms_left-1)/5)
        rem = terms_left - 1
        lb = target - Fraction(rem, 5)
        if lb > 0:
            d_max = lb.denominator // lb.numerator
        else:
            d_max = 10000000
        d_max = min(d_max, 10000000)
        
        if d_max < d_start:
            return None
        
        current_set = set(current)
        
        for d in range(d_start, d_max + 1):
            if d in current_set:
                continue
            if time.time() > deadline:
                return None
            
            new_target = target - Fraction(1, d)
            if new_target < 0:
                break
            
            # Prune: new_target must be <= max_sum_from(d+1, terms_left-1)
            max_rem = max_sum_from(d + 1, terms_left - 1)
            if new_target > max_rem:
                continue
            
            result = search(new_target, d + 1, terms_left - 1, current + [d])
            if result is not None:
                return result
        
        return None
    
    for k in range(9, 16):
        if time.time() > deadline - 5.0:
            break
        result = search(Fraction(1, 1), 5, k, [])
        if result is not None:
            s = sum(Fraction(1, d) for d in result)
            assert s == 1
            assert len(result) == k
            assert all(5 <= d <= 10000000 for d in result)
            assert len(set(result)) == len(result)
            return result
    
    # Fallback
    fallback = [5, 6, 7, 8, 9, 10, 11, 13, 15, 20, 30, 42, 65, 110, 195]
    s = sum(Fraction(1, d) for d in fallback)
    if s == 1:
        return fallback
    raise RuntimeError("no representation found")
