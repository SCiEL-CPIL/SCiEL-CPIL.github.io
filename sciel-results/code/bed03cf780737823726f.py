"""Bootstrap portfolio: MODEL-AND-OPTIMIZE for Egyptian fraction of 5/31."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256
DEADLINE = 60.0  # safety margin

def _try_set(denoms):
    """Verify a set of denominators sums to TARGET and meets constraints."""
    if not denoms:
        return False
    if len(denoms) > MAX_TERMS:
        return False
    s = sum(Fraction(1, d) for d in denoms)
    if s != TARGET:
        return False
    if len(set(denoms)) != len(denoms):
        return False
    if any(d < 1 or d > MAX_DENOM for d in denoms):
        return False
    return True

def _bounded_complete(remaining: Fraction, last_d: int, used: set, max_allowed: int, depth: int = 0):
    """Recursively complete the Egyptian fraction with all denominators <= max_allowed.
    Returns list of denominators or None if no valid completion exists."""
    if remaining == 0:
        return []
    if depth >= MAX_TERMS:
        return None
    if remaining < 0:
        return None
    
    # The next denominator must be > last_d and <= max_allowed
    # Also, 1/d <= remaining, so d >= ceil(1/remaining)
    min_d = max(last_d + 1, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
    if min_d > max_allowed:
        return None
    
    # Upper bound: we need enough remaining terms to sum to 'remaining'
    # With (MAX_TERMS - depth) terms left, each at most 1/min_d, 
    # but we need sum = remaining. 
    # A quick check: if (MAX_TERMS - depth) * (1/min_d) < remaining, impossible
    # But that's a loose bound. Better: the maximum sum achievable with k terms all >= min_d
    # is k/min_d. If k/min_d < remaining, impossible.
    k = MAX_TERMS - depth
    if k * Fraction(1, min_d) < remaining:
        return None
    
    # Try denominators from min_d upward, but limit the range for efficiency
    # The greedy choice is d0 = ceil(1/remaining). Try a window around it.
    d0 = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
    # Try d from min_d to min(max_allowed, d0 + 50) but at least d0
    start_d = min_d
    end_d = min(max_allowed, d0 + 100)
    
    for d in range(start_d, end_d + 1):
        if d in used:
            continue
        new_rem = remaining - Fraction(1, d)
        if new_rem < 0:
            continue
        used.add(d)
        result = _bounded_complete(new_rem, d, used, max_allowed, depth + 1)
        used.discard(d)
        if result is not None:
            return [d] + result
    return None

def _greedy_complete(remaining: Fraction, current_max: int, used: set):
    """Complete the Egyptian fraction greedily, ensuring distinct denominators.
    Returns list of denominators or None if fails."""
    result = []
    r = remaining
    last_d = current_max
    while r > 0:
        if r.numerator == 0:
            break
        d = (r.denominator + r.numerator - 1) // r.numerator
        while d in used or d <= last_d:
            d += 1
        if d > MAX_DENOM:
            return None
        if len(result) + len(used) >= MAX_TERMS:
            return None
        r -= Fraction(1, d)
        result.append(d)
        last_d = d
    if r != 0:
        return None
    return result

def construct():
    start_time = time.time()
    best_result = None
    best_max_denom = float('inf')
    
    # Get baseline greedy solution
    r = TARGET
    baseline = []
    last_d = 0
    while r > 0:
        d = (r.denominator + r.numerator - 1) // r.numerator
        if d <= last_d:
            d = last_d + 1
        baseline.append(d)
        r -= Fraction(1, d)
        last_d = d
    if _try_set(baseline):
        best_result = baseline[:]
        best_max_denom = max(baseline)
    
    # Strategy: Binary search on max_allowed using bounded recursive completion
    # Try to find the smallest M such that there exists a valid Egyptian fraction
    # of 5/31 with all denominators <= M
    
    # First, get a good upper bound from greedy
    upper_bound = best_max_denom if best_result else MAX_DENOM
    lower_bound = 7  # minimum possible (5/31 < 1/6, so at least 2 terms, min denom > 6)
    
    # Binary search for the minimum achievable max denominator
    lo = lower_bound
    hi = upper_bound
    
    # We'll try bounded search for various M values
    # Start with the current best and try to improve
    candidate_ms = []
    # Try values between lower_bound and upper_bound
    # Use a logarithmic spacing to cover the range efficiently
    if upper_bound > lower_bound:
        # Generate candidate M values
        import math
        steps = 20
        for i in range(steps + 1):
            m = int(lower_bound + (upper_bound - lower_bound) * i / steps)
            candidate_ms.append(m)
        # Also add some specific values
        for m in range(lower_bound, min(upper_bound + 1, lower_bound + 50)):
            if m not in candidate_ms:
                candidate_ms.append(m)
        candidate_ms = sorted(set(candidate_ms))
    
    # Try each candidate M from smallest to largest
    for M in candidate_ms:
        if time.time() - start_time > DEADLINE - 3:
            break
        if M >= best_max_denom:
            break
        
        # Try to find a solution with all denominators <= M
        # Try different first denominators
        found = False
        for first_d in range(7, M + 1):
            if time.time() - start_time > DEADLINE - 3:
                found = False
                break
            rem = TARGET - Fraction(1, first_d)
            if rem < 0:
                continue
            if rem == 0:
                if first_d <= M:
                    result = [first_d]
                    if _try_set(result):
                        best_max_denom = M
                        best_result = result
                        found = True
                break
            
            used = {first_d}
            comp = _bounded_complete(rem, first_d, used, M)
            if comp is not None:
                result = [first_d] + comp
                if _try_set(result):
                    best_max_denom = M
                    best_result = result
                    found = True
                    break
        
        if found:
            # We found a solution with max <= M, try smaller M
            continue
    
    # If we didn't find anything better, try bounded search with the current best
    if best_result and time.time() - start_time < DEADLINE - 5:
        # Try to improve by searching with slightly smaller M
        for M in range(max(7, best_max_denom - 30), best_max_denom):
            if time.time() - start_time > DEADLINE - 3:
                break
            for first_d in range(7, M + 1):
                if time.time() - start_time > DEADLINE - 3:
                    break
                rem = TARGET - Fraction(1, first_d)
                if rem < 0:
                    continue
                if rem == 0:
                    if _try_set([first_d]):
                        best_max_denom = M
                        best_result = [first_d]
                    break
                used = {first_d}
                comp = _bounded_complete(rem, first_d, used, M)
                if comp is not None:
                    result = [first_d] + comp
                    if _try_set(result):
                        best_max_denom = M
                        best_result = result
                        break
    
    if best_result is None:
        # Fallback: pure greedy
        r = TARGET
        result = []
        last_d = 0
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d <= last_d:
                d = last_d + 1
            result.append(d)
            r -= Fraction(1, d)
            last_d = d
        return result
    
    return best_result
# EVOLVE-BLOCK-END
