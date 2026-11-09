"""Bootstrap portfolio: MODEL-AND-OPTIMIZE for Egyptian fraction of 5/31."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256
DEADLINE = 55.0  # Safety margin for 60s limit

def _validate(denoms):
    """Validate a list of denominators. Return max if valid, else None."""
    if not denoms or len(denoms) > MAX_TERMS:
        return None
    if len(set(denoms)) != len(denoms):
        return None
    if any(d < 1 or d > MAX_DENOM for d in denoms):
        return None
    total = sum(Fraction(1, d) for d in denoms)
    if total != TARGET:
        return None
    return max(denoms)

def _greedy_complete(remaining, used, start_d, deadline):
    """Complete the Egyptian fraction greedily. Returns list or None."""
    result = []
    r = remaining
    last_d = start_d
    while r > 0:
        if time.time() > deadline:
            return None
        d = (r.denominator + r.numerator - 1) // r.numerator
        while d in used or d <= last_d:
            d += 1
            if d > MAX_DENOM:
                return None
        if d > MAX_DENOM:
            return None
        result.append(d)
        used.add(d)
        r -= Fraction(1, d)
        last_d = d
        if len(result) > MAX_TERMS:
            return None
    return result

def _search_bounded(rem, used, min_d, limit_max, deadline, depth=0):
    """
    Bounded DFS: find Egyptian fraction of 'rem' with all denominators in [min_d, limit_max),
    not in 'used'. Returns list of denominators or None.
    """
    if time.time() > deadline:
        return None
    if rem == 0:
        return []
    if depth >= MAX_TERMS:
        return None
    if rem < 0:
        return None
    
    # Lower bound: need 1/d <= rem for positive remainder (or handle multi-term)
    # Actually for the first term we need d >= ceil(1/rem) to keep remainder non-negative
    # But we might use multiple terms. The key constraint: d must be >= min_d and < limit_max
    min_d_eff = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
    if min_d_eff >= limit_max:
        return None
    
    # Upper bound on d: if we pick d, remainder is rem - 1/d.
    # We need rem - 1/d >= 0, so d >= ceil(1/rem) which is min_d_eff.
    # Also, if rem - 1/d > 0, we need to represent it with remaining terms all < limit_max.
    
    for d in range(min_d_eff, limit_max):
        if d in used:
            continue
        rem_new = rem - Fraction(1, d)
        if rem_new < 0:
            break
        if rem_new == 0:
            return [d]
        res = _search_bounded(rem_new, used | {d}, d + 1, limit_max, deadline, depth + 1)
        if res is not None:
            return [d] + res
    return None

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Known solution: [8, 31, 248], max = 248
    best_solution = [8, 31, 248]
    best_max = _validate(best_solution)
    if best_max is None:
        best_max = 248  # fallback
    
    # Try to find solutions with max < best_max
    # We'll try decreasing limit_max from best_max-1 downward
    # For each limit, search for any valid representation
    
    for limit in range(best_max - 1, 20, -1):
        if time.time() > deadline:
            break
        
        # Try different first denominators
        # d1 must satisfy: 1/d1 < 5/31 => d1 >= 7
        # d1 must be < limit
        for d1 in range(7, limit):
            if time.time() > deadline:
                break
            r1 = TARGET - Fraction(1, d1)
            if r1 <= 0:
                continue
            
            # Try to complete r1 with denominators in [d1+1, limit)
            used = {d1}
            res = _search_bounded(r1, used, d1 + 1, limit, deadline)
            if res is not None:
                full_sol = [d1] + res
                mx = _validate(full_sol)
                if mx is not None and mx < best_max:
                    best_max = mx
                    best_solution = full_sol
                    # Don't break - try to find even better
                    # But if we find something very good, we can stop
                    if best_max <= 31:
                        return best_solution
    
    # If bounded search didn't improve, try greedy completion from various starting points
    for d1 in range(7, best_max):
        if time.time() > deadline:
            break
        r1 = TARGET - Fraction(1, d1)
        if r1 <= 0:
            continue
        used = {d1}
        completion = _greedy_complete(r1, used, d1, deadline)
        if completion is not None:
            full_sol = [d1] + completion
            mx = _validate(full_sol)
            if mx is not None and mx < best_max:
                best_max = mx
                best_solution = full_sol
    
    return best_solution
# EVOLVE-BLOCK-END
