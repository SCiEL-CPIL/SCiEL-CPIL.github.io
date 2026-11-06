from fractions import Fraction
import time
import math

_DEADLINE = 0
_BEST = None

def construct():
    global _DEADLINE, _BEST
    start_time = time.time()
    _DEADLINE = start_time + 55  # 55s safety margin before 90s kill
    
    # Try k from 9 upward (7,8 are impossible as shown)
    # k=9 is the minimum possible
    for k in range(9, 30):
        if time.time() > _DEADLINE:
            break
        result = search_optimized(Fraction(1, 1), 5, k)
        if result is not None:
            return result
    
    # Fallback: greedy solution
    return greedy_solution()

def search_optimized(target, min_denom, terms_left):
    """
    Optimized DFS with strong pruning.
    """
    global _DEADLINE
    
    if time.time() > _DEADLINE:
        return None
    
    if terms_left == 0:
        return [] if target == 0 else None
    
    if terms_left == 1:
        if target.numerator != 1:
            return None
        d = target.denominator
        if d >= min_denom and d <= 10000000:
            return [d]
        return None
    
    if terms_left == 2:
        # 1/a + 1/b = target, a < b, a >= min_denom
        # 1/a = target - 1/b, so a = 1/(target - 1/b)
        # Or: b = a / (a*target - 1), and b > a
        # a*target - 1 must divide a, and b = a/(a*target - 1) > a
        # So a*target - 1 < 1, meaning a*target < 2
        # Also a*target - 1 > 0, so a > 1/target
        # And b = a/(a*target - 1), need b > a => 1/(a*target - 1) > 1 => a*target - 1 < 1 => a*target < 2
        # So we need: 1/target < a < 2/target, and (a*target - 1) divides a
        # Let g = gcd(a, a*target - 1). Since gcd(a, a*target - 1) = gcd(a, -1) = 1... 
        # Wait, that's not right. Let me redo.
        # 1/a + 1/b = target => b = a / (a*target - 1)
        # For b to be integer, (a*target - 1) must divide a
        # Let d = a*target - 1, then b = a/d, so d | a
        # a = d*b, and d = a*target - 1 = d*b*target - 1
        # d + 1 = d*b*target
        # d(1 - b*target) = -1
        # d(b*target - 1) = 1
        # So b*target - 1 = 1/d for some integer d... this is getting complicated.
        # 
        # Alternative: iterate a from min_denom, check if b is integer and > a
        # b = a / (a*target - 1)
        # Need a*target - 1 > 0 and a*target - 1 | a
        # a*target - 1 > 0 => a > 1/target
        # a*target - 1 < a (since b > a means 1/(a*target-1) > 1 means a*target - 1 < 1)
        # So 0 < a*target - 1 < 1, but a*target - 1 is not necessarily integer...
        # 
        # Let me just iterate. The range is small.
        a_min = min_denom
        if target.numerator > 0:
            inv = Fraction(1, 1) / target
            a_min = max(a_min, inv.numerator // inv.denominator + 1)
        
        # a*target < 2 => a < 2/target
        a_max = 10000000
        if target.numerator > 0:
            two_over_target = Fraction(2, 1) / target
            a_max = min(a_max, two_over_target.numerator // two_over_target.denominator)
        
        for a in range(a_min, a_max + 1):
            if time.time() > _DEADLINE:
                return None
            # b = a / (a*target - 1)
            num = a
            den = a * target - 1
            if den <= 0:
                continue
            if num % den != 0:
                continue
            b = num // den
            if b > a and b >= min_denom and b <= 10000000:
                return [a, b]
        return None
    
    # Pruning: max possible sum
    max_sum = Fraction(0, 1)
    for i in range(terms_left):
        max_sum += Fraction(1, min_denom + i)
    if max_sum < target:
        return None
    
    # Lower bound for d
    d_min = min_denom
    if target.numerator > 0:
        inv_target = Fraction(1, 1) / target
        d_from_target = inv_target.numerator // inv_target.denominator
        if inv_target.numerator % inv_target.denominator != 0:
            d_from_target += 1
        d_min = max(d_min, d_from_target)
    
    # Upper bound: d such that sum_{i=0}^{terms_left-1} 1/(d+i) >= target
    # Approximate: terms_left/d >= target => d <= terms_left/target
    d_max_approx = int(terms_left / target) + 3
    d_max = min(10000000, max(d_min, d_max_approx))
    
    # Additional pruning: if d is too large, even the max sum of remaining terms is too small
    for d in range(d_min, d_max + 1):
        if time.time() > _DEADLINE:
            return None
        
        term = Fraction(1, d)
        if term > target:
            break
        
        new_target = target - term
        if new_target <= 0:
            continue
        
        # Check if new_target can be formed by terms_left-1 terms > d
        # Max sum of remaining
        max_rest = Fraction(0, 1)
        for i in range(terms_left - 1):
            max_rest += Fraction(1, d + 1 + i)
        
        if new_target > max_rest:
            break
        
        # Min sum check: new_target must be > 0 (already checked)
        # Also, new_target must be at least the sum of the smallest possible remaining terms
        # But that's 0 in the limit, so no useful lower bound
        
        res = search_optimized(new_target, d + 1, terms_left - 1)
        if res is not None:
            return [d] + res
    
    return None

def greedy_solution():
    """Fallback greedy solution."""
    R = Fraction(1, 1)
    D = 5
    denoms = []
    while R > 0:
        inv_R = Fraction(R.denominator, R.numerator)
        d = max(D, inv_R.numerator // inv_R.denominator)
        if (inv_R.numerator % inv_R.denominator) != 0:
            d += 1
        if d < D:
            d = D
        denoms.append(d)
        R -= Fraction(1, d)
        D = d + 1
        if len(denoms) > 256:
            break
    return denoms
