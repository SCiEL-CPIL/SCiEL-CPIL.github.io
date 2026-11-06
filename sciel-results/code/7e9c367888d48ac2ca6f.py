"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

_MAX_DEN = 10000000


def construct():
    """
    Find the minimum number of distinct denominators in [5, 10000000]
    whose reciprocal sum is exactly 1.
    """
    deadline = time.time() + 55.0
    
    for k in range(8, 257):
        if time.time() > deadline - 2.0:
            break
        sol = _search_k(k, deadline)
        if sol is not None:
            # Verify the solution
            total_num, total_den = _sum_reciprocals(sol)
            if total_num == total_den:
                return sol
    
    # Fallback: return a known valid solution (larger count)
    return _fallback()


def _sum_reciprocals(denoms):
    """Compute sum of 1/d for d in denoms as (num, den) reduced fraction."""
    from math import gcd
    num, den = 0, 1
    for d in denoms:
        # num/den + 1/d = (num*d + den) / (den*d)
        new_num = num * d + den
        new_den = den * d
        g = gcd(new_num, new_den)
        if g > 1:
            new_num //= g
            new_den //= g
        num, den = new_num, new_den
    return num, den


def _search_k(k, deadline):
    """
    Backtracking search for exactly k distinct denominators >= 5
    whose reciprocal sum is exactly 1.
    Returns list of denominators or None.
    """
    
    def rec(remaining, last, num, den):
        """
        Find 'remaining' more denominators > last such that
        num/den + sum(1/d_i) = 1, i.e., sum(1/d_i) = (den-num)/den.
        """
        if time.time() > deadline:
            return None
        
        target_num = den - num
        if target_num <= 0:
            return None
        
        if remaining == 0:
            return None
        
        if remaining == 1:
            # Need 1/d = target_num/den => d = den/target_num
            if den % target_num == 0:
                d = den // target_num
                if d > last and d >= 5 and d <= _MAX_DEN:
                    return [d]
            return None
        
        # Lower bound on each denominator: 1/d <= target_num/den => d >= den/target_num
        # So d_low = ceil(den / target_num)
        d_low = (den + target_num - 1) // target_num
        if d_low < last + 1:
            d_low = last + 1
        if d_low < 5:
            d_low = 5
        
        # Upper bound: even with the smallest possible remaining denominators,
        # we need enough sum. The max sum of 'remaining' terms with denominators
        # starting from d_low is approximately sum(1/(d_low + i) for i in range(remaining))
        # But more tightly: the first term must be at most target_num/(remaining * ... )
        # Actually, 1/d1 <= target_num/den, so d1 >= d_low (already have this)
        # And we need sum of remaining-1 terms to be <= target_num/den - 1/d1
        # The minimum sum of remaining-1 terms with denominators > d1 is 
        # sum(1/(d1+1 + i) for i in range(remaining-1))
        # If target_num/den - 1/d1 < that minimum sum, no solution.
        
        # Also, the maximum sum of 'remaining' terms with denominators >= d_low
        # is sum(1/(d_low + i) for i in range(remaining))
        # If target_num/den > that, no solution.
        max_possible = _max_sum_float(d_low, remaining)
        if target_num / den > max_possible + 1e-15:
            return None
        
        # The minimum sum of 'remaining' terms with denominators >= d_low
        # is sum(1/(d_low + remaining - 1 + i) for i in range(remaining))
        # i.e., the largest possible denominators... but they're unbounded above
        # So the minimum sum approaches 0. Not useful for pruning.
        
        # Better: for the first denominator d1, we need:
        # 1/d1 < target_num/den (strictly, since we need more terms)
        # and target_num/den - 1/d1 must be representable as sum of (remaining-1) unit fractions
        # with denominators > d1.
        # The max sum of (remaining-1) terms with denominators > d1 is 
        # sum(1/(d1+1+i) for i in range(remaining-1))
        # So we need target_num/den - 1/d1 <= that max.
        # => 1/d1 >= target_num/den - max_sum(d1+1, remaining-1)
        
        d = d_low
        guard = 0
        while d <= _MAX_DEN:
            guard += 1
            if guard > 500000 or time.time() > deadline:
                return None
            
            # Check: 1/d < target_num/den (strictly, need more terms)
            if d * target_num >= den * 1:
                # 1/d >= target_num/den, too big or equal (equal only if remaining==1, handled)
                break
            
            # Check: target_num/den - 1/d must be <= max sum of (remaining-1) terms > d
            # target_num/den - 1/d = (target_num * d - den) / (den * d)
            rem_num = target_num * d - den
            rem_den = den * d
            if rem_num <= 0:
                break
            
            # Reduce
            g = math.gcd(rem_num, rem_den)
            if g > 1:
                rem_num //= g
                rem_den //= g
            
            # Max possible sum of (remaining-1) terms with denominators > d
            max_rem = _max_sum_float(d + 1, remaining - 1)
            if rem_num / rem_den > max_rem + 1e-15:
                # Even the max sum of remaining terms is too small
                break
            
            # Min possible: we need at least (remaining-1) distinct denominators > d
            # The minimum sum approaches 0, so no lower bound pruning here.
            # But we need rem_num/rem_den > 0, already checked.
            
            res = rec(remaining - 1, d, rem_num, rem_den)
            if res is not None:
                return [d] + res
            
            d += 1
        
        return None
    
    return rec(k, 4, 0, 1)


def _max_sum_float(start, count):
    """Approximate max reciprocal sum with `count` distinct denominators >= start."""
    s = 0.0
    for j in range(count):
        s += 1.0 / (start + j)
    return s


def _fallback():
    """
    Return a valid (possibly non-optimal) solution.
    Use the identity: 1/n = 1/(n+1) + 1/(n*(n+1))
    Start: 1 = 1/5 + 4/5
    4/5 = 1/6 + 19/30
    19/30 = 1/7 + 103/210
    103/210 = 1/8 + 614/1680 = 1/8 + 307/840
    307/840 = 1/9 + 1923/7560 = 1/9 + 641/2520
    641/2520 = 1/10 + 3890/25200 = 1/10 + 389/2520
    389/2520 = 1/11 + 1759/27720
    1759/27720 = 1/16 + 424/443520 = 1/16 + 53/55440
    53/55440 = 1/54 + ... this gets complicated.
    
    Let me just use a known valid expansion computed programmatically.
    """
    # Use greedy-like: repeatedly split
    # 1 = 1/5 + 4/5
    # 4/5: 1/6, rem 19/30
    # 19/30: 1/7, rem 103/210
    # 103/210: 1/8, rem 307/840
    # 307/840: 1/9, rem 641/2520
    # 641/2520: 1/10, rem 389/2520
    # 389/2520: 1/11, rem 1759/27720
    # 1759/27720: 1/16, rem 53/55440
    # 53/55440: 1/54, rem 53/55440 - 1/54 = 53*54/55440*54 - 55440/55440*54... 
    # Let me just compute it properly.
    
    from fractions import Fraction
    
    remaining = Fraction(1, 1)
    denoms = []
    last = 4
    
    while remaining > 0:
        # Find smallest d > last such that 1/d <= remaining
        d = last + 1
        while Fraction(1, d) > remaining:
            d += 1
        if d > _MAX_DEN:
            break
        denoms.append(d)
        remaining -= Fraction(1, d)
        last = d
        if remaining == 0:
            break
    
    if remaining == 0 and len(denoms) <= 256:
        return denoms
    
    # This shouldn't happen, but as a last resort
    raise RuntimeError("no solution found")
# EVOLVE-BLOCK-END
