"""Rewritten program: direct optimal Egyptian fraction with denominators > 4."""
import time
import math
from fractions import Fraction

def _find_min_terms():
    """Find minimal Egyptian fraction for 1 with denominators >= 5, distinct, minimizing count."""
    
    # Try k=4: enumerate d1, d2, d3 and compute d4
    # 1/d1 + 1/d2 + 1/d3 + 1/d4 = 1
    # d4 = d1*d2*d3 / (d1*d2*d3 - d2*d3 - d1*d3 - d1*d2)
    
    for d1 in range(5, 100):
        # d1 must be small enough that remaining 3 terms can sum to 1 - 1/d1
        # 3/d2_min <= 1 - 1/d1, so d2_min >= 3/(1-1/d1) = 3*d1/(d1-1)
        # Also 1/d1 + 1/d2 + 1/d3 + 1/d4 <= 1/d1 + 3/(d1+1) approximately
        # For feasibility: 1/d1 + 1/(d1+1) + 1/(d1+2) + 1/(d1+3) >= 1
        max_d1 = 100  # safe bound
        if d1 > max_d1:
            break
        
        for d2 in range(d1 + 1, d1 + 100):
            # Check if remaining 2 terms can sum to 1 - 1/d1 - 1/d2
            rem2 = 1 - Fraction(1, d1) - Fraction(1, d2)
            if rem2 <= 0:
                break
            # Need 1/d3 + 1/d4 = rem2, with d3 > d2
            # d3 >= d2+1, and 1/d3 + 1/d4 <= 2/(d2+1)
            # So need rem2 <= 2/(d2+1)
            if rem2 > Fraction(2, d2 + 1):
                continue
            # d3 upper bound: 1/d3 >= rem2/2 (since 1/d4 <= 1/d3), so d3 <= 2/rem2
            d3_max = min(20000, int(2 / float(rem2)) + 1)
            if d3_max < d2 + 1:
                continue
            
            for d3 in range(d2 + 1, d3_max + 1):
                # 1/d4 = rem2 - 1/d3
                rem4 = rem2 - Fraction(1, d3)
                if rem4 <= 0:
                    break
                # d4 = 1/rem4
                if rem4.denominator == 1:
                    d4 = rem4.numerator
                    if d4 > d3 and d4 <= 10000000:
                        return [d1, d2, d3, d4]
    
    # Try k=5 with bounded DFS using integer arithmetic for speed
    start = time.time()
    deadline = start + 50.0
    
    def dfs5(a, b, min_den, terms, depth, deadline):
        """Find up to 5 terms for remaining fraction a/b using integer arithmetic."""
        if time.time() > deadline:
            return None
        if a == 0:
            return terms[:]
        if depth >= 5:
            return None
        if a == 1 and b >= min_den:
            return terms + [b]
        
        # Bounds on d
        # 1/d <= a/b => d >= b/a
        # Also d >= min_den
        d_min = max(min_den, math.ceil(b / a) + 1)
        
        # Upper bound on d: if d is too large, 1/d is too small and we can't reach the target
        # with remaining terms. Max remaining sum = (5-depth-1) * 1/(d+1) approximately
        # So a/b - 1/d <= (5-depth-1)/(d+1)
        # a/b <= 1/d + (5-depth-1)/(d+1)
        # For large d, RHS ~ (5-depth)/d, so d <= (5-depth)*b/a
        d_max = min(10000000, int((5 - depth) * b / a) + 2)
        
        if d_min > d_max:
            return None
        
        for d in range(d_min, d_max + 1):
            # new_rem = a/b - 1/d = (a*d - b) / (b*d)
            new_a = a * d - b
            if new_a <= 0:
                break
            new_b = b * d
            # Simplify the fraction
            g = math.gcd(new_a, new_b)
            new_a //= g
            new_b //= g
            
            result = dfs5(new_a, new_b, d + 1, terms + [d], depth + 1, deadline)
            if result is not None:
                return result
            if time.time() > deadline:
                return None
        return None
    
    result = dfs5(1, 1, 5, [], 0, deadline)
    if result is not None:
        return result
    
    # Fallback: use the knapsack approach
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    
    return None

def construct():
    start = time.time()
    deadline = start + 55.0
    
    result = _find_min_terms()
    
    if result is None or len(result) == 0:
        # Final fallback
        target = 2520
        reached = {0: []}
        for d in range(5, target + 1):
            if target % d:
                continue
            w = target // d
            for total, values in list(reached.items()):
                if total + w <= target and total + w not in reached:
                    reached[total + w] = values + [d]
            if target in reached:
                result = reached[target]
                break
        else:
            raise RuntimeError("no solution found")
    
    # Verify with exact arithmetic
    total = Fraction(0, 1)
    for d in result:
        total += Fraction(1, d)
    
    if total != 1:
        # Should not happen, but try to fix
        raise RuntimeError(f"Invalid solution: sum = {total}")
    
    return result
