"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm as _lcm

_MAX_DEN = 10000000


def _construct():
    """
    Find the minimum number of distinct denominators in [5, 10^7]
    whose reciprocal sum is exactly 1.
    
    Uses integer-arithmetic backtracking with strong pruning.
    """
    deadline = time.time() + 75.0  # safety margin before 90s kill
    
    # For each k, try to find k distinct denominators summing to 1
    # We search k from 8 upward (k<8 is impossible)
    
    for k in range(8, 257):
        if time.time() > deadline:
            break
        sol = _search_k(k, deadline)
        if sol is not None:
            return sol
    
    # Fallback: guaranteed valid representation via DP
    return _fallback()


def _search_k(k, deadline):
    """
    Backtracking search for exactly k distinct denominators >= 5
    whose reciprocal sum is 1, using integer arithmetic.
    
    State: (num, den) = remaining fraction to fill (as reduced fraction)
    We maintain the invariant that num/den is the remaining sum.
    """
    # We'll search for denominators d1 < d2 < ... < dk
    # such that 1/d1 + ... + 1/dk = 1, all di >= 5
    
    # Use a list to track chosen denominators
    chosen = []
    
    def rec(remaining, last, num, den):
        """
        Find `remaining` distinct denominators > last such that
        their reciprocal sum equals num/den.
        """
        if time.time() > deadline:
            return False
        
        if remaining == 0:
            return num == 0
        
        if remaining == 1:
            # Need 1/d = num/den, so d = den/num must be integer
            if num == 0:
                return False
            if den % num == 0:
                d = den // num
                if d > last and d >= 5 and d <= _MAX_DEN:
                    chosen.append(d)
                    return True
            return False
        
        # remaining >= 2
        # Next denominator d must satisfy:
        #   1/d < num/den  =>  d > den/num
        #   d > last
        #   d >= 5
        # Also, the sum of the largest `remaining` reciprocals starting after d
        # must be >= num/den (pruning)
        
        # Lower bound on d
        d_low = max(last + 1, 5)
        if num > 0:
            # 1/d < num/den means d > den/num
            d_low = max(d_low, den // num + 1)
        
        # Upper bound on d: we need enough remaining terms to fill num/den
        # The max sum with `remaining` terms starting from d is bounded by
        # the sum of 1/d + 1/(d+1) + ... + 1/(d+remaining-1)
        # But we need this to be >= num/den for feasibility.
        # A simpler bound: 1/d >= num/(den * remaining) is not quite right.
        # Better: if 1/d < num/den, that's already enforced by d_low.
        # For upper bound: we need remaining terms each <= 1/d (since they're larger denominators)
        # So max possible sum is remaining/d. We need remaining/d >= num/den
        # => d <= remaining * den / num
        if num > 0:
            d_high = min(_MAX_DEN, (remaining * den) // num)
        else:
            d_high = _MAX_DEN
        
        if d_low > d_high:
            return False
        
        # Try each candidate d in [d_low, d_high]
        for d in range(d_low, d_high + 1):
            if time.time() > deadline:
                return False
            
            # Check: 1/d < num/den  (i.e., den < num * d)
            if not (den < num * d):
                continue
            
            # New remaining: num/den - 1/d = (num*d - den) / (den*d)
            new_num = num * d - den
            new_den = den * d
            
            # Reduce the fraction
            if new_num == 0:
                # We've exactly filled it, but we still need more terms
                # This can only work if remaining == 1, but we're in remaining >= 2
                continue
            
            g = _gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            # Pruning: check if the max possible sum of (remaining-1) terms
            # with denominators > d can fill new_num/new_den
            # Max sum ≈ (remaining-1)/d (rough upper bound)
            # More precise: sum of 1/(d+1) + ... + 1/(d+remaining-1)
            # Use harmonic approximation for speed
            if new_num > 0:
                # Upper bound on sum of next (remaining-1) terms:
                # Each term <= 1/(d+1), so sum <= (remaining-1)/(d+1)
                # We need (remaining-1)/(d+1) >= new_num/new_den
                # => (remaining-1) * new_den >= new_num * (d+1)
                if (remaining - 1) * new_den < new_num * (d + 1):
                    break  # larger d only makes this worse
            
            chosen.append(d)
            if rec(remaining - 1, d, new_num, new_den):
                return True
            chosen.pop()
        
        return False
    
    # Start: we need sum = 1/1, i.e., num=1, den=1
    if rec(k, 4, 1, 1):
        return chosen[:]
    return None


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _fallback():
    """
    Guaranteed-valid baseline: subset DP over divisors of a common denominator.
    Returns a valid (if not minimal) set of denominators.
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
    return _construct()
# EVOLVE-BLOCK-END
