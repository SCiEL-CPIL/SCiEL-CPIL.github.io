"""Egyptian fraction of 5/31 with minimal largest denominator."""
import time
from fractions import Fraction
from math import gcd, isqrt

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256

def _verify(denoms):
    """Verify a set of denominators sums to TARGET and meets constraints."""
    if not denoms:
        return False
    if len(denoms) > MAX_TERMS:
        return False
    if len(set(denoms)) != len(denoms):
        return False
    if any(d < 1 or d > MAX_DENOM for d in denoms):
        return False
    s = sum(Fraction(1, d) for d in denoms)
    return s == TARGET

def _search_3_terms_diophantine(max_limit):
    """
    Search for 3-term Egyptian fraction solutions 1/a + 1/b + 1/c = 5/31
    where max(a,b,c) < max_limit.
    Uses the identity: if c is fixed, (Ka - 31Mc)(Kb - 31Mc) = (31Mc)^2
    where K = 5c - 31.
    """
    best_solution = None
    best_max = max_limit
    
    # Iterate c (the largest denominator) downwards from max_limit - 1
    for c in range(max_limit - 1, 1, -1):
        K = 5 * c - 31
        if K <= 0:
            continue
        
        # We need to solve 1/a + 1/b = K / (31c)
        # (Ka - 31cK)(Kb - 31cK) = (31cK)^2
        # Let X = Ka - 31cK, Y = Kb - 31cK
        # XY = (31cK)^2
        # a = (X + 31cK) / K
        # b = (Y + 31cK) / K
        
        # For a, b to be integers, X must be a divisor of (31cK)^2
        # such that (X + 31cK) is divisible by K.
        # Note: 31cK is divisible by K, so X must be congruent to 0 mod K?
        # No. X + 31cK ≡ X (mod K). So X must be a multiple of K.
        # Let X = K * u. Then K * u * Y = (31cK)^2 => u * Y = (31cK)^2 / K = 31c * 31c * K.
        # So u must be a divisor of 31c * 31c * K.
        # Also a = (Ku + 31cK) / K = u + 31c.
        # b = (Y + 31cK) / K. Since Y = (31c * 31c * K) / u,
        # b = (31c * 31c * K / u + 31cK) / K = 31c * 31c / u + 31c = 31c(31c/u + 1).
        # For b to be integer, u must divide 31c * 31c.
        
        # So we iterate divisors u of N = 31c * 31c.
        N = 31 * c * 31 * c
        # u must be such that a = u + 31c and b = 31c * (31c/u + 1) are valid.
        # We want distinct a, b, c.
        # WLOG a <= b < c.
        
        # Iterate divisors u of N
        # u < sqrt(N) for a <= b
        sqrt_N = isqrt(N)
        
        for u in range(1, sqrt_N + 1):
            if N % u != 0:
                continue
            
            a = u + 31 * c
            # Check if a < c (since a <= b < c)
            if a >= c:
                # If a >= c, then b >= a >= c, so max would be >= c.
                # But we are looking for solutions with max < max_limit.
                # If we fix c as the largest, we need a < c and b < c.
                # If a >= c, this branch doesn't yield a solution where c is strictly largest 
                # in the sense that a,b < c. 
                # However, if a == c, denominators are not distinct.
                # If a > c, then a is the largest. But we are iterating c as the candidate largest.
                # So we require a < c.
                continue
                
            v = N // u
            b = 31 * c * (v // (31 * c) + 1) if (31 * c * v) % (31 * c) == 0 else None
            # Simplify b calculation: b = 31c * (31c/u + 1) = 31c * (31c + u) / u
            # Since u divides 31c * 31c, let's check divisibility carefully.
            # b = 31c * (N/u + 1) / (31c) ? No.
            # b = 31c * (31c/u + 1).
            # For b to be integer, 31c/u must be integer? No.
            # b = (31c * 31c + 31c * u) / u = 31c * (31c + u) / u.
            # We know u | 31c * 31c.
            # Let 31c * 31c = u * w.
            # b = 31c * (w + 1).
            # This is always an integer.
            w = N // u
            b = 31 * c * (w + 1)
            
            if b <= 0:
                continue
            
            # Check distinctness
            if a == b or a == c or b == c:
                continue
            
            # Check order a <= b < c
            if a > b:
                a, b = b, a
            if b >= c:
                continue
            
            # Check if max(a,b,c) < max_limit
            # Since b < c, max is c.
            # We are iterating c < max_limit, so this is satisfied.
            
            # Verify the solution
            if _verify([a, b, c]):
                # Update best if c is smaller than current best_max
                # We are iterating c downwards, so the first valid c is the smallest possible
                # for 3-term solutions?
                # Yes, we start from max_limit - 1 and go down.
                # The first one we find has the smallest c among those < max_limit.
                return [a, b, c], c
        
    return None, max_limit

def _search_n_terms(n_terms, max_denom_limit, time_budget):
    """
    Fallback recursive search for n_terms > 3.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    best_result = None
    best_max = max_denom_limit
    
    def backtrack(remaining_terms, remaining_fraction, min_denom, current_denoms):
        nonlocal best_result, best_max
        
        if time.time() > deadline:
            return
        
        if remaining_terms == 0:
            if remaining_fraction == 0:
                max_d = max(current_denoms)
                if max_d < best_max:
                    best_max = max_d
                    best_result = current_denoms[:]
            return
        
        if remaining_fraction <= 0:
            return
        
        lower = max(min_denom, int(1 / remaining_fraction) + 1)
        if lower < 1:
            lower = 1
        
        if remaining_terms == 1:
            if remaining_fraction.numerator == 1:
                d = remaining_fraction.denominator
                if d >= min_denom and d < best_max and d <= MAX_DENOM:
                    candidate = current_denoms + [d]
                    if len(set(candidate)) == len(candidate):
                        best_max = d
                        best_result = candidate[:]
            return
        
        upper = int(remaining_terms / remaining_fraction) + 1
        if upper > MAX_DENOM:
            upper = MAX_DENOM
        
        for d in range(lower, upper + 1):
            if time.time() > deadline:
                return
            
            if d >= best_max:
                break
            
            if d in current_denoms:
                continue
            
            new_remaining = remaining_fraction - Fraction(1, d)
            if new_remaining <= 0:
                continue
            
            min_possible = (remaining_terms - 1) / Fraction(d + 1)
            if new_remaining < min_possible:
                break
            
            max_possible = (remaining_terms - 1) / Fraction(d)
            if new_remaining > max_possible:
                continue
            
            current_denoms.append(d)
            backtrack(remaining_terms - 1, new_remaining, d + 1, current_denoms)
            current_denoms.pop()
        
        return
    
    backtrack(n_terms, TARGET, 7, [])
    
    if best_result is not None:
        return best_result, best_max
    return None, max_denom_limit

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Known solution: 5/31 = 1/8 + 1/31 + 1/248
    known = [8, 31, 248]
    if _verify(known):
        pass
    
    best_result = known[:]
    best_max_denom = 248
    
    # Try 2-term solutions (quick check, though we know none exist)
    if time.time() < deadline - 5:
        result, max_d = _search_n_terms(2, best_max_denom, 5)
        if result is not None and max_d < best_max_denom:
            best_result = result
            best_max_denom = max_d
    
    # Try 3-term solutions using Diophantine method
    # This is much faster than recursion
    if time.time() < deadline - 10 and best_max_denom > 50:
        result, max_d = _search_3_terms_diophantine(best_max_denom)
        if result is not None and max_d < best_max_denom:
            best_result = result
            best_max_denom = max_d
    
    # Try 4-term solutions if there's time and potential for improvement
    if time.time() < deadline - 15 and best_max_denom > 30:
        result, max_d = _search_n_terms(4, best_max_denom, 30)
        if result is not None and max_d < best_max_denom:
            best_result = result
            best_max_denom = max_d
            
    # Try 5-term solutions if there's time and potential for improvement
    if time.time() < deadline - 20 and best_max_denom > 20:
        result, max_d = _search_n_terms(5, best_max_denom, 40)
        if result is not None and max_d < best_max_denom:
            best_result = result
            best_max_denom = max_d

    return best_result
