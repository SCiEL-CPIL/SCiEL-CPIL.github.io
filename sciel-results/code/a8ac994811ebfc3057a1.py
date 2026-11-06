import time
from fractions import Fraction
from math import gcd

# EVOLVE-BLOCK-START

def _find_min_terms(n_terms, deadline):
    """Find n_terms distinct denominators in [5, 10^7] with reciprocal sum = 1."""
    from fractions import Fraction
    
    def dfs(start, remaining, chosen):
        if time.time() > deadline:
            return None
        if len(chosen) == n_terms:
            if remaining == Fraction(0, 1):
                return list(chosen)
            return None
        if remaining <= 0:
            return None
        
        # We need (n_terms - len(chosen)) more terms
        terms_left = n_terms - len(chosen)
        
        # The maximum possible sum of remaining terms: 1/start + 1/(start+1) + ... 
        # The minimum possible: each term at most 1/(start + terms_left - 1) roughly
        # For pruning: remaining must be <= sum of largest possible remaining terms
        # and >= sum of smallest possible remaining terms
        
        # Upper bound on what we can add: terms starting from 'start'
        # Max sum = 1/start + 1/(start+1) + ... + 1/(start+terms_left-1)
        max_possible = sum(Fraction(1, start + i) for i in range(terms_left))
        if remaining > max_possible:
            return None
        
        # Min sum = terms starting from as large as possible, but they must be distinct and > start
        # Actually min is when we pick the largest possible denominators, but that's unbounded
        # Better lower bound: each remaining term >= 1/(some large number), so min sum approaches 0
        # But we need remaining > 0, already checked.
        
        # Also: the smallest denominator we can pick is 'start'
        # The largest denominator for the first of 'terms_left' terms:
        # If we pick d as the next term, then 1/d <= remaining, so d >= 1/remaining
        # Also d >= start
        # And we need the remaining (terms_left - 1) terms to sum to remaining - 1/d
        # with all > d
        
        # Lower bound on d: d >= max(start, ceil(1/remaining))
        # But also, the remaining (terms_left-1) terms must sum to remaining - 1/d > 0
        # So 1/d < remaining, i.e., d > 1/remaining
        
        # Upper bound on d: we need to be able to make up remaining - 1/d with (terms_left-1) terms all > d
        # Max sum of (terms_left-1) terms starting from d+1: 1/(d+1) + ... + 1/(d+terms_left-1)
        # So remaining - 1/d <= 1/(d+1) + ... + 1/(d+terms_left-1)
        
        d_min = max(start, 1)
        if remaining < 1:
            d_min = max(d_min, (1 // remaining.numerator) * remaining.denominator + 1)
            # 1/d < remaining => d > 1/remaining
            # 1/remaining = remaining.denominator / remaining.numerator
            d_min = max(d_min, remaining.denominator // remaining.numerator + 1)
        
        # Upper bound: d must be small enough that remaining terms can fill the gap
        # remaining - 1/d >= minimum possible from (terms_left-1) terms > d
        # Min is approaching 0, so no tight lower bound on d from this
        # But d can't be too large because 1/d must be <= remaining
        
        # Let's iterate d from d_min upward
        # Upper bound on d: 1/d <= remaining => d >= 1/remaining (already have)
        # Also d < remaining.denominator / remaining.numerator + 1 roughly
        
        # For efficiency, cap d
        # If terms_left == 1: d = 1/remaining must be integer
        if terms_left == 1:
            if remaining.numerator == 1:
                d = remaining.denominator
                if d >= start and d <= 10000000:
                    return list(chosen) + [d]
            elif remaining.denominator % remaining.numerator == 0:
                d = remaining.denominator // remaining.numerator
                if d >= start and d <= 10000000:
                    return list(chosen) + [d]
            return None
        
        # For terms_left >= 2:
        # d must satisfy: 1/d < remaining (so remaining - 1/d > 0)
        # and remaining - 1/d <= max sum of (terms_left-1) terms from d+1
        # and remaining - 1/d >= min sum (approaches 0)
        
        # Upper bound on d: 
        # remaining - 1/d <= 1/(d+1) + 1/(d+2) + ... + 1/(d+terms_left-1)
        # For large d, RHS ≈ (terms_left-1)/d, and LHS ≈ remaining
        # So remaining ≈ (terms_left-1)/d => d ≈ (terms_left-1)/remaining
        # This gives an upper bound
        
        d_max_est = int((terms_left - 1) / remaining) + start + 100
        d_max = min(d_max_est, 10000000)
        
        for d in range(d_min, d_max + 1):
            if time.time() > deadline:
                return None
            if d < start:
                continue
            frac_d = Fraction(1, d)
            if frac_d >= remaining:
                continue
            new_remaining = remaining - frac_d
            result = dfs(d + 1, new_remaining, chosen + [d])
            if result is not None:
                return result
        return None
    
    return dfs(5, Fraction(1, 1), [])

def _search_minimum(deadline):
    """Search for the minimum number of terms, trying 5, 6, 7, ... in order."""
    # We know 4 terms is impossible (max sum < 1)
    # Try 5 terms first
    for n in range(5, 15):
        if time.time() > deadline:
            break
        sol = _find_min_terms(n, deadline)
        if sol is not None:
            return sol
    return None

def construct():
    """Return 1..256 distinct integer denominators in [5,10000000] whose reciprocal sum is exactly 1; minimize their count."""
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s limit
    
    # Search for minimum term count
    sol = _search_minimum(deadline)
    if sol is not None:
        # Validate
        total = sum(Fraction(1, d) for d in sol)
        if total == 1 and len(set(sol)) == len(sol):
            if all(5 <= d <= 10000000 for d in sol):
                if 1 <= len(sol) <= 256:
                    return sol
    
    # Fallback: use a known valid solution
    # Let me construct one: use the greedy algorithm to get a valid (possibly non-minimal) solution
    remaining = Fraction(1, 1)
    chosen = []
    d = 5
    while remaining > 0 and len(chosen) < 256:
        # Pick the smallest d >= 5 not in chosen such that 1/d < remaining
        # and we can still complete
        found = False
        for dd in range(max(d, 5), 10000001):
            if dd in set(chosen):
                continue
            frac = Fraction(1, dd)
            if frac < remaining:
                # Check if remaining - frac can be completed with denominators > dd
                # For now, just take it if it seems reasonable
                new_rem = remaining - frac
                if new_rem > 0:
                    chosen.append(dd)
                    remaining = new_rem
                    d = dd + 1
                    found = True
                    break
        if not found:
            break
    
    if remaining == 0 and 1 <= len(chosen) <= 256:
        total = sum(Fraction(1, x) for x in chosen)
        if total == 1:
            return chosen
    
    # Last resort: hardcoded valid solution
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30
    # This sums to more than 1. Let me find a valid one.
    # Use: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30
    # Sum ≈ 2.01, too much.
    
    # Let me use a different approach: find any valid solution
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30
    # No, let me just return a known valid set.
    
    # Actually, let me try a specific known decomposition:
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30
    # This doesn't work. Let me just raise.
    
    raise RuntimeError("no solution found")

# EVOLVE-BLOCK-END
