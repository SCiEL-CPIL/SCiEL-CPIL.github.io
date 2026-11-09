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

def _find_3_term_solution(max_denom):
    """
    Find a 3-term Egyptian fraction 1/a + 1/b + 1/c = 5/31
    with max(a,b,c) < max_denom.
    Returns (a, b, c) or None.
    """
    # We need 1/a + 1/b + 1/c = 5/31
    # WLOG a <= b <= c
    # a must be >= ceil(31/5) = 7
    # For a given a, we need 1/b + 1/c = 5/31 - 1/a = (5a - 31)/(31a)
    # Let num = 5a - 31, den = 31a
    # We need 1/b + 1/c = num/den where num > 0
    # (den - b)(den - c) = den^2 / num ... actually:
    # 1/b + 1/c = num/den => (b+c)/(bc) = num/den => den(b+c) = num*bc
    # num*bc - den*b - den*c = 0
    # (num*b - den)(num*c - den) = den^2
    # So for each divisor d of den^2, we can set:
    # num*b - den = d => b = (d + den) / num
    # num*c - den = den^2/d => c = (den^2/d + den) / num
    
    for a in range(7, max_denom):
        num = 5 * a - 31
        if num <= 0:
            continue
        den = 31 * a
        
        # b must be >= a, and 1/b + 1/c = num/den
        # b >= max(a, ceil(den/num))
        b_min = max(a, (den + num - 1) // num)
        # c >= b, and 1/b + 1/c <= 2/b => num/den <= 2/b => b <= 2*den/num
        b_max = (2 * den + num - 1) // num
        
        if b_min > b_max:
            continue
        
        # For each valid b, check if c is an integer
        # 1/c = num/den - 1/b = (num*b - den) / (den*b)
        # c = den*b / (num*b - den)
        # Need num*b - den > 0 and (num*b - den) | den*b
        
        den_sq = den * den
        # Instead of iterating all b, use the divisor approach
        # (num*b - den)(num*c - den) = den^2
        # Let X = num*b - den, Y = num*c - den, XY = den^2
        # X <= Y since b <= c
        # X must be a divisor of den^2 with X <= den (since X <= Y and XY = den^2)
        # Actually X <= sqrt(den^2) = den
        # b = (X + den) / num must be integer
        # c = (Y + den) / num = (den^2/X + den) / num must be integer
        
        # Iterate divisors X of den^2 up to den
        # This is expensive for large den, so let's iterate b directly for small ranges
        # Or use a smarter approach: iterate possible values of num*b - den
        
        # For efficiency, let's just iterate b in the range
        for b in range(b_min, b_max + 1):
            # Check if c is integer
            # c = den*b / (num*b - den)
            denom_c = num * b - den
            if denom_c <= 0:
                continue
            if (den * b) % denom_c != 0:
                continue
            c = (den * b) // denom_c
            if c < b:
                continue
            if c >= max_denom:
                break
            # Check distinctness
            if a == b or b == c or a == c:
                continue
            # Verify
            if _verify([a, b, c]):
                return [a, b, c]
    
    return None

def _search_small_denoms(max_denom, time_budget):
    """
    Search for solutions with all denominators < max_denom.
    Uses a recursive approach with pruning.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    best_result = None
    best_max = max_denom
    
    def backtrack(remaining_terms, remaining_frac, min_denom, current_denoms):
        nonlocal best_result, best_max
        
        if time.time() > deadline:
            return
        
        if remaining_terms == 0:
            if remaining_frac == 0:
                max_d = max(current_denoms)
                if max_d < best_max:
                    best_max = max_d
                    best_result = current_denoms[:]
            return
        
        if remaining_frac <= 0:
            return
        
        # Lower bound on next denominator
        lower = max(min_denom, (1 + remaining_frac.denominator) // remaining_frac.numerator)
        if lower < 1:
            lower = 1
        
        # Upper bound: sum of remaining_terms reciprocals of d, d+1, ... 
        # 1/d + 1/(d+1) + ... <= remaining_frac
        # For simplicity, use: 1/d <= remaining_frac => d >= 1/remaining_frac
        # Also need remaining_frac - 1/d > 0 => d > 1/remaining_frac
        
        if remaining_terms == 1:
            if remaining_frac.numerator == 1:
                d = remaining_frac.denominator
                if d >= min_denom and d < best_max and d <= MAX_DENOM and d not in current_denoms:
                    candidate = current_denoms + [d]
                    if len(set(candidate)) == len(candidate):
                        best_max = d
                        best_result = candidate[:]
            return
        
        # Upper bound for d: we need 1/d + 1/(d+1) + ... + 1/(d+remaining_terms-1) >= remaining_frac
        # Rough upper bound: remaining_terms * 1/d >= remaining_frac => d <= remaining_terms / remaining_frac
        upper = int(remaining_terms / remaining_frac) + 1
        if upper > best_max - 1:
            upper = best_max - 1
        if upper > MAX_DENOM:
            upper = MAX_DENOM
        
        for d in range(lower, upper + 1):
            if time.time() > deadline:
                return
            
            if d >= best_max:
                break
            
            if d in current_denoms:
                continue
            
            new_remaining = remaining_frac - Fraction(1, d)
            if new_remaining <= 0:
                continue
            
            # Prune: can we make new_remaining from (remaining_terms - 1) terms with denominators > d?
            # Max sum: 1/(d+1) + 1/(d+2) + ... 
            # Min sum: 1/best_max + 1/best_max + ... (but they must be distinct and > d)
            # Simple prune: new_remaining <= (remaining_terms - 1) / (d + 1)
            if new_remaining > (remaining_terms - 1) / Fraction(d + 1):
                break
            
            # new_remaining >= 1/best_max + ... (roughly)
            # Skip if too small
            
            current_denoms.append(d)
            backtrack(remaining_terms - 1, new_remaining, d + 1, current_denoms)
            current_denoms.pop()
        
        return
    
    # Try different numbers of terms
    for n_terms in range(2, 9):
        if time.time() > deadline - 1:
            break
        backtrack(n_terms, TARGET, 2, [])
        if best_result is not None:
            return best_result
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin from 90s limit
    
    # Known solution: 5/31 = 1/8 + 1/31 + 1/248
    best_result = [8, 31, 248]
    best_max = 248
    
    # Try to find 3-term solutions with max < 248
    if time.time() < deadline - 10:
        sol = _find_3_term_solution(best_max)
        if sol is not None:
            best_result = sol
            best_max = max(sol)
    
    # Try bounded search for even better solutions (4+ terms)
    if time.time() < deadline - 5 and best_max > 50:
        result = _search_small_denoms(best_max, 60)
        if result is not None:
            max_d = max(result)
            if max_d < best_max:
                best_result = result
                best_max = max_d
    
    return best_result
