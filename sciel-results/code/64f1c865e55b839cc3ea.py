"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit
    
    target = Fraction(5, 31)
    
    # We know [8, 31, 248] works: 1/8 + 1/31 + 1/248 = 5/31
    # Let's try to find a solution with a smaller maximum denominator.
    
    # Strategy: Use meet-in-the-middle for small term counts.
    # Precompute all possible sums of k unit fractions with denominators in [1, C]
    # and check if target - sum is achievable by the complementary set.
    
    # But first, let's try a direct search for 2-term and 3-term solutions
    # with smaller maximum denominators.
    
    # 2-term: 1/a + 1/b = 5/31
    # 31(a+b) = 5ab => 5ab - 31a - 31b = 0
    # (5a - 31)(5b - 31) = 31^2 = 961
    # 961 = 1 * 961 = 11 * 87.45... wait, 961 = 31^2
    # Factors of 961: 1, 31, 961
    # 5a - 31 = d, 5b - 31 = 961/d
    # a = (d + 31)/5, b = (961/d + 31)/5
    
    # d must divide 961 and d+31 must be divisible by 5.
    # d = 1: a = 32/5 (not integer)
    # d = 31: a = 62/5 (not integer)
    # d = 961: a = 992/5 (not integer)
    # So no 2-term solution exists.
    
    # 3-term: 1/a + 1/b + 1/c = 5/31
    # We can search over a and b, then solve for c.
    # 1/c = 5/31 - 1/a - 1/b = (5ab - 31b - 31a) / (31ab)
    # c = 31ab / (5ab - 31a - 31b)
    # For c to be a positive integer, we need 5ab - 31a - 31b > 0 and (5ab - 31a - 31b) | 31ab
    
    # Let's search for 3-term solutions with max denom < 248.
    # We'll iterate a from 1 to 247, b from a+1 to 247, and check if c is integer and c > b.
    
    found = None
    
    for a in range(1, 248):
        for b in range(a + 1, 248):
            num = 31 * a * b
            den = 5 * a * b - 31 * a - 31 * b
            if den <= 0:
                continue
            if num % den != 0:
                continue
            c = num // den
            if c <= b:
                continue
            if c > 247:
                continue
            # Verify
            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                found = [a, b, c]
                break
        if found:
            break
        if time.time() > deadline - 15:
            break
    
    if found:
        return sorted(found)
    
    # If no 3-term solution with max denom < 248, let's try 4-term solutions.
    # This is more complex. We'll use a recursive search with pruning.
    
    def find_solution(rem, max_denom, min_denom, exclude, depth):
        """
        Find a representation of rem as sum of unit fractions.
        All denominators in [min_denom, max_denom], not in exclude.
        Returns list of denominators or None.
        """
        if rem <= 0 or depth == 0:
            return None
        
        p = rem.numerator
        q = rem.denominator
        
        # Check if rem is a unit fraction
        if q % p == 0:
            d = q // p
            if min_denom <= d <= max_denom and d not in exclude:
                return [d]
        
        # Bounds for the next denominator
        # 1/d < rem => d > q/p
        # Also, depth * (1/d) >= rem => d <= depth * q / p
        
        min_d = max(min_denom, q // p + 1)
        max_d = min(max_denom, depth * q // p)
        
        if min_d > max_d:
            return None
        
        for d in range(min_d, max_d + 1):
            if d in exclude:
                continue
            
            new_rem = rem - Fraction(1, d)
            if new_rem <= 0:
                continue
            
            result = find_solution(new_rem, max_denom, d + 1, exclude | {d}, depth - 1)
            if result is not None:
                return [d] + result
        
        return None
    
    # Try to find a 4-term solution with max denom < 248.
    # We'll try C from 247 down to 200, with a time limit.
    
    for C in range(247, 199, -1):
        if time.time() > deadline - 5:
            break
        result = find_solution(target, C, 1, set(), 4)
        if result is not None:
            return sorted(result)
    
    # If we couldn't find anything smaller, return the known solution 248.
    return [8, 31, 248]
# EVOLVE-BLOCK-END
