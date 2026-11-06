"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

_MAX_DEN = 10000000

def _max_fill(last, count):
    """
    Calculate the maximum sum of 'count' distinct reciprocal fractions
    with denominators strictly greater than 'last'.
    This is the sum of 1/(last+1) + ... + 1/(last+count).
    Returns (numerator, denominator) of the sum.
    """
    if count == 0:
        return (0, 1)
    
    num = 0
    den = 1
    for j in range(1, count + 1):
        d = last + j
        # Add 1/d to num/den
        # new_num = num * d + den
        # new_den = den * d
        # We can reduce at each step to keep numbers small, but for small counts
        # it's fine to accumulate and reduce at the end or periodically.
        # Given count <= 256, numbers can get large. Let's reduce periodically.
        num = num * d + den
        den = den * d
        # Reduce fraction
        g = gcd(num, den)
        if g > 1:
            num //= g
            den //= g
    return (num, den)

def _search(k, deadline):
    """
    Backtracking search for exactly k distinct denominators >= 5
    whose reciprocal sum is 1.
    """
    def rec(remaining, last, num, den, chosen):
        """
        remaining: number of terms left to choose
        last: the previous denominator chosen
        num/den: the fraction remaining to be summed (num/den = 1 - sum(chosen))
        chosen: list of chosen denominators so far
        """
        if time.time() > deadline:
            return None
        
        if remaining == 0:
            if num == 0 and den == 1:
                return chosen[:]
            return None
        
        if remaining == 1:
            # We need 1/d = num/den => d = den/num
            if num == 0:
                return None
            if den % num == 0:
                d = den // num
                if d > last and d >= 5 and d <= _MAX_DEN:
                    return chosen + [d]
            return None
        
        # remaining >= 2
        # Lower bound for next denominator d:
        # 1. d > last
        # 2. 1/d < num/den  => d > den/num
        d_low = max(last + 1, den // num + 1)
        
        # Upper bound for next denominator d:
        # The sum of the remaining 'remaining' terms must be at least num/den.
        # The maximum sum of 'remaining' terms starting from d is bounded by
        # sum_{i=0}^{remaining-1} 1/(d+i).
        # If this max sum < num/den, then d is too large.
        # Also, the first term 1/d must be <= num/den (already handled by d_low).
        # A simple upper bound: d cannot be so large that even the smallest
        # possible sum of the remaining terms (which is very small) isn't the issue,
        # but rather that the *maximum* possible sum is insufficient.
        
        # Let's iterate d from d_low upwards.
        # We can stop if the max possible sum of 'remaining' terms starting at d
        # is less than num/den.
        
        d = d_low
        # Guard against infinite loops or excessive iterations
        guard = 0
        while d <= _MAX_DEN:
            guard += 1
            if guard > 5000000 or time.time() > deadline:
                return None
            
            # Check if 1/d < num/den
            # den < num * d
            if not (den < num * d):
                d += 1
                continue
            
            # New remaining fraction: num/den - 1/d = (num*d - den) / (den*d)
            new_num = num * d - den
            new_den = den * d
            
            # Reduce
            if new_num == 0:
                # This would mean we filled the sum exactly with 1 term, 
                # but we have 'remaining' > 1 terms left. 
                # This branch is invalid because we need to place more terms.
                # However, if remaining was 1, we handled it above.
                # So if new_num == 0 and remaining > 1, it's a dead end for this d?
                # No, it means 1/d = num/den. But we need 'remaining' terms.
                # If we pick d, we've used 1 term. We have remaining-1 terms left to sum to 0.
                # This is impossible unless remaining-1 == 0, which is not the case.
                # So we can skip this d.
                d += 1
                continue
            
            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            # Pruning: Check if the max sum of (remaining - 1) terms starting from d+1
            # can cover new_num/new_den.
            # Max sum = sum_{j=1}^{remaining-1} 1/(d+j)
            # If max_sum < new_num/new_den, then break (larger d makes max_sum even smaller)
            if remaining - 1 > 0:
                max_fill_num, max_fill_den = _max_fill(d, remaining - 1)
                # Check if max_fill_num/max_fill_den < new_num/new_den
                # <=> max_fill_num * new_den < new_num * max_fill_den
                if max_fill_num * new_den < new_num * max_fill_den:
                    break
            
            chosen.append(d)
            res = rec(remaining - 1, d, new_num, new_den, chosen)
            if res is not None:
                return res
            chosen.pop()
            
            d += 1
        
        return None

    return rec(k, 4, 1, 1, [])

def _fallback():
    """
    Guaranteed-valid baseline.
    Returns a valid set of denominators, not necessarily minimal.
    """
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            new_total = total + w
            if new_total <= target and new_total not in reached:
                reached[new_total] = values + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no baseline representation")

def construct():
    # k < 8 is impossible.
    # We search for the minimum k.
    deadline = time.time() + 80.0  # Safety margin
    
    for k in range(8, 257):
        if time.time() > deadline:
            break
        sol = _search(k, deadline)
        if sol is not None:
            return sol
    
    return _fallback()
# EVOLVE-BLOCK-END
