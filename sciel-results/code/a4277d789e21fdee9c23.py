"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

_MAX_DEN = 10000000


def _construct():
    """
    Find the minimum number of distinct denominators in [5, 10^7]
    whose reciprocal sum is exactly 1.
    """
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # Try increasing k from 5 upward
    for k in range(5, 257):
        if time.time() > deadline:
            break
        sol = _search_k(k, deadline)
        if sol is not None:
            return sol
    
    # Fallback: known valid representation
    return _fallback()


def _search_k(k, deadline):
    """
    Backtracking search for exactly k distinct denominators >= 5
    whose reciprocal sum is 1, using integer arithmetic.
    """
    chosen = []
    
    def rec(remaining, last, num, den):
        """
        Find `remaining` distinct denominators > last such that
        their reciprocal sum equals num/den (in reduced form).
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
        #   1/d <= num/den  =>  d >= den/num (but strictly: 1/d < num/den if we need more terms)
        #   Actually for the last term we need 1/d = num/den exactly
        #   For non-last terms, we need 1/d < num/den (strictly less, since remaining > 1)
        #   d > last
        #   d >= 5
        
        # Lower bound: d > den/num (since 1/d < num/den => d > den/num)
        # But also d > last and d >= 5
        d_low = max(last + 1, 5)
        if num > 0:
            # 1/d < num/den => d > den/num
            # So d >= floor(den/num) + 1
            d_low = max(d_low, den // num + 1)
        
        # Upper bound: we need enough terms to fill num/den
        # The maximum sum of `remaining` terms with denominators >= d is 
        # 1/d + 1/(d+1) + ... + 1/(d+remaining-1) < remaining/d
        # We need this >= num/den
        # So remaining/d >= num/den => d <= remaining * den / num
        if num > 0:
            d_high = min(_MAX_DEN, (remaining * den) // num)
        else:
            d_high = _MAX_DEN
        
        if d_low > d_high:
            return False
        
        # Additional pruning: the sum of the smallest possible remaining terms
        # (using the largest denominators) must still be achievable
        # But more useful: try candidates in order
        
        for d in range(d_low, d_high + 1):
            if time.time() > deadline:
                return False
            
            # Check: 1/d < num/den (i.e., den < num * d)
            if den >= num * d:
                continue
            
            # New remaining: num/den - 1/d = (num*d - den) / (den*d)
            new_num = num * d - den
            new_den = den * d
            
            if new_num <= 0:
                continue
            
            # Reduce the fraction
            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            # Pruning: check if max possible sum of (remaining-1) terms with denominators > d
            # can fill new_num/new_den
            # Max sum <= (remaining-1)/(d+1)
            # Need (remaining-1)/(d+1) >= new_num/new_den
            # => (remaining-1) * new_den >= new_num * (d+1)
            if (remaining - 1) * new_den < new_num * (d + 1):
                break  # larger d makes this worse
            
            # Also: min possible sum with (remaining-1) terms starting from d+1
            # is approximately (remaining-1)/(d+remaining-1) (very rough)
            # Skip this for now
            
            chosen.append(d)
            if rec(remaining - 1, d, new_num, new_den):
                return True
            chosen.pop()
        
        return False
    
    # Start: sum = 1/1, so num=1, den=1
    if rec(k, 4, 1, 1):
        return chosen[:]
    return None


def _fallback():
    """
    Guaranteed-valid baseline representation.
    """
    # Known: 1 = 1/5 + 1/6 + 1/7 + 1/42 + 1/140 + 1/280 + 1/840
    # Verify: 1/5 + 1/6 + 1/7 + 1/42 + 1/140 + 1/280 + 1/840
    # LCM of 5,6,7,42,140,280,840 = 840
    # 840/5=168, 840/6=140, 840/7=120, 840/42=20, 840/140=6, 840/280=3, 840/840=1
    # Sum = 168+140+120+20+6+3+1 = 458... that's not 840. Let me recompute.
    
    # Let me use a verified identity:
    # 1 = 1/2 + 1/3 + 1/7 + 1/42 + 1/1806 + 1/3263442 + 1/10854616880 (standard greedy)
    # But all must be >= 5.
    
    # Alternative: use the fact that 1/5 + 1/6 + 1/7 + 1/42 = 1 - 1/140? No.
    # 1/5+1/6+1/7+1/42 = (42+35+30+10)/210 = 117/210 = 39/70. Not 1.
    
    # Let me just use a DP-based fallback with a moderate target.
    target = 2520
    # Find subset of divisors of target (>= 5) that sum to target
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
