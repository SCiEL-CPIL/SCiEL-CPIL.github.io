"""Egyptian fraction of 5/31 with minimal largest denominator."""
import time
from fractions import Fraction
from math import gcd

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

def _search_n_terms(n_terms, max_denom_limit, time_budget):
    """
    Search for Egyptian fraction representations with exactly n_terms.
    Returns (denoms, max_denom) for best solution found, or (None, infinity).
    Uses recursive backtracking with tight bounds.
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
        
        # Lower bound on next denominator:
        # 1/d <= remaining_fraction, so d >= 1/remaining_fraction
        # Also d >= min_denom
        # Upper bound: we need remaining_terms-1 more terms, each >= 1/(d+1)
        # So (remaining_terms-1) * 1/(d+1) <= remaining_fraction - 1/d
        # This gives a tighter bound
        
        lower = max(min_denom, int(1 / remaining_fraction) + 1)
        if lower < 1:
            lower = 1
        
        # For the last term, we can solve directly
        if remaining_terms == 1:
            if remaining_fraction.numerator == 1:
                d = remaining_fraction.denominator
                if d >= min_denom and d < best_max and d <= MAX_DENOM:
                    candidate = current_denoms + [d]
                    if len(set(candidate)) == len(candidate):
                        best_max = d
                        best_result = candidate[:]
            return
        
        # For remaining_terms >= 2, bound the search
        # d must satisfy: remaining_fraction - 1/d > 0 (so d > 1/remaining_fraction)
        # And we need (remaining_terms-1) terms to sum to remaining_fraction - 1/d
        # Each of those terms is at least 1/(d+1), so:
        # (remaining_terms-1)/(d+1) <= remaining_fraction - 1/d
        # This gives an upper bound on d
        
        # Simple upper bound: d < 1/remaining_fraction * remaining_terms
        # because 1/d + (remaining_terms-1)/d = remaining_terms/d <= remaining_fraction
        # so d >= remaining_terms / remaining_fraction
        # Wait, that's a lower bound. Let me think again.
        # We have remaining_terms terms each <= 1/d (since they're >= d)
        # So remaining_fraction <= remaining_terms / d
        # Thus d <= remaining_terms / remaining_fraction
        upper = int(remaining_terms / remaining_fraction) + 1
        if upper > MAX_DENOM:
            upper = MAX_DENOM
        
        for d in range(lower, upper + 1):
            if time.time() > deadline:
                return
            
            if d >= best_max:
                break
            
            # Check if d is already used
            if d in current_denoms:
                continue
            
            new_remaining = remaining_fraction - Fraction(1, d)
            if new_remaining <= 0:
                continue
            
            # Prune: if new_remaining < (remaining_terms-1) / (d+1), impossible
            # because the smallest possible sum of (remaining_terms-1) terms each >= 1/(d+1)
            min_possible = (remaining_terms - 1) / Fraction(d + 1)
            if new_remaining < min_possible:
                break
            
            # Prune: if new_remaining > (remaining_terms-1) / d, impossible
            # because the largest possible sum of (remaining_terms-1) terms each <= 1/d
            max_possible = (remaining_terms - 1) / Fraction(d)
            if new_remaining > max_possible:
                continue
            
            current_denoms.append(d)
            backtrack(remaining_terms - 1, new_remaining, d + 1, current_denoms)
            current_denoms.pop()
        
        return
    
    # Start with no denominators, minimum denominator 1
    # But we know denominators must be at least 7 (since 1/6 > 5/31)
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
    
    # Try 2-term solutions first (fastest)
    if time.time() < deadline - 5:
        result, max_d = _search_n_terms(2, best_max_denom, 5)
        if result is not None and max_d < best_max_denom:
            best_result = result
            best_max_denom = max_d
    
    # Try 3-term solutions
    if time.time() < deadline - 10 and best_max_denom > 50:
        result, max_d = _search_n_terms(3, best_max_denom, 20)
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
