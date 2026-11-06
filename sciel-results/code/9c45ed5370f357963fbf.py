from fractions import Fraction
import time
import math

_DEADLINE = 0

def construct():
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + 50  # safety margin before 90s kill

    # Minimum possible count is 9.
    # We search for k=9 first. If found, return it.
    # If not found in time, we try k=10, etc., or fallback to greedy.
    # Given the constraints, a 9-term solution is expected to exist and be findable.
    
    # Try to find a 9-term solution
    res = _search(1, 1, 5, 9)
    if res is not None:
        return res

    # If 9 terms not found quickly (should not happen with good pruning), try 10
    if time.time() < _DEADLINE - 5:
        res = _search(1, 1, 5, 10)
        if res is not None:
            return res

    # Fallback: guaranteed-valid greedy (may use more terms, but always valid).
    return _greedy()

def _search(num, den, min_denom, terms_left):
    """
    Find `terms_left` distinct denominators >= min_denom, each <= 10^7,
    whose reciprocal sum equals num/den exactly.
    Returns list of denominators or None.
    """
    global _DEADLINE

    if time.time() > _DEADLINE:
        return None

    if terms_left == 0:
        return [] if num == 0 else None

    if terms_left == 1:
        if num == 0:
            return None
        if den % num != 0:
            return None
        d = den // num
        if d >= min_denom and d <= 10000000:
            return [d]
        return None

    if terms_left == 2:
        # Solve 1/d1 + 1/d2 = num/den
        # 1/d2 = (num*d1 - den) / (den*d1)
        # d2 = den*d1 / (num*d1 - den)
        # We need d2 > d1 and d2 integer.
        # d2 > d1 => den*d1 / (num*d1 - den) > d1
        # den > num*d1 - den  (since num*d1 - den > 0 for valid d1)
        # 2*den > num*d1 => d1 < 2*den/num
        # Also 1/d1 < num/den => d1 > den/num
        d_min = min_denom
        lb = den // num
        if den % num != 0:
            lb += 1
        d_min = max(d_min, lb)
        
        d_max = min(10000000, int((2 * den) // num))
        
        for d1 in range(d_min, d_max + 1):
            if time.time() > _DEADLINE:
                return None
            
            num_d1 = num * d1
            if num_d1 <= den:
                continue
                
            denom_d2 = num_d1 - den
            numer_d2 = den * d1
            
            if numer_d2 % denom_d2 != 0:
                continue
                
            d2 = numer_d2 // denom_d2
            if d2 > d1 and d2 <= 10000000:
                return [d1, d2]
                
        return None

    # Prune: max possible sum with `terms_left` terms starting at min_denom.
    max_sum = 0.0
    for i in range(terms_left):
        max_sum += 1.0 / (min_denom + i)
    if max_sum < num / den - 1e-15:
        return None

    # Bounds on d:
    d_min = min_denom
    if num > 0:
        lb = den // num
        if den % num != 0:
            lb += 1
        d_min = max(d_min, lb)

    # Upper bound: terms_left / d >= num/den => d <= terms_left * den / num
    d_max = min(10000000, int((terms_left * den) // max(num, 1)) + 5)
    if d_max < d_min:
        return None

    for d in range(d_min, d_max + 1):
        if time.time() > _DEADLINE:
            return None

        # 1/d must be < target
        if num * d >= den:
            continue

        n2 = num * d - den
        d2 = den * d

        # Prune: max sum of remaining (terms_left-1) terms starting at d+1
        # must be >= n2/d2.
        # Approx check for speed
        max_rest = 0.0
        for i in range(terms_left - 1):
            max_rest += 1.0 / (d + 1 + i)
        if max_rest < n2 / d2 - 1e-15:
            break

        res = _search(n2, d2, d + 1, terms_left - 1)
        if res is not None:
            return [d] + res

    return None

def _greedy():
    """Guaranteed valid fallback (Egyptian greedy), denominators >= 5."""
    R = Fraction(1, 1)
    D = 5
    denoms = []
    while R > 0 and len(denoms) < 256:
        # d = ceil(1/R)
        d = (R.denominator + R.numerator - 1) // R.numerator
        if d < D:
            d = D
        denoms.append(d)
        R -= Fraction(1, d)
        D = d + 1
    return denoms
