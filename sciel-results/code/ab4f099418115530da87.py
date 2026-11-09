"""Bootstrap portfolio: MODEL-AND-OPTIMIZE for Egyptian fraction of 5/31."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256
DEADLINE = 85.0  # safety margin (evaluator kills at 90s)

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

def construct():
    start_time = time.time()
    
    # Known optimal: 5/31 = 1/8 + 1/31 + 1/248
    # Verify: 1/8 + 1/31 + 1/248 = (31 + 8 + 1)/248 = 40/248 = 5/31
    known = [8, 31, 248]
    if _verify(known):
        return known
    
    # Fallback: search for a valid solution
    best_result = None
    best_max_denom = float('inf')
    
    # Try greedy with various first denominators
    for first_d in range(6, 51):
        if time.time() - start_time > DEADLINE - 2:
            break
        rem = TARGET - Fraction(1, first_d)
        if rem < 0:
            continue
        if rem == 0:
            continue
        used = {first_d}
        result = [first_d]
        r = rem
        last_d = first_d
        success = True
        while r > 0:
            if time.time() - start_time > DEADLINE - 2:
                success = False
                break
            d = (r.denominator + r.numerator - 1) // r.numerator
            while d <= last_d or d in used:
                d += 1
            if d > MAX_DENOM:
                success = False
                break
            if len(result) >= MAX_TERMS:
                success = False
                break
            r -= Fraction(1, d)
            result.append(d)
            last_d = d
        if success and r == 0 and _verify(result):
            max_d = max(result)
            if max_d < best_max_denom:
                best_max_denom = max_d
                best_result = result[:]
    
    if best_result is not None:
        return best_result
    
    # Final fallback: pure greedy
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
# EVOLVE-BLOCK-END
