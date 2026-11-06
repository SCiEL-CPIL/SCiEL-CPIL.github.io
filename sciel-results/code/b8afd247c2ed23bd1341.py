"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd as _gcd

_MAX_DEN = 10000000


def _construct():
    """
    Find the minimum number of distinct denominators in [5, 10^7]
    whose reciprocal sum is exactly 1.
    """
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # k=7 is impossible (max sum of 7 smallest: 1/5+...+1/11 < 1)
    # k=8: max sum of 8 smallest > 1, so feasible in principle
    for k in range(8, 257):
        if time.time() > deadline:
            break
        sol = _search_k(k, deadline)
        if sol is not None:
            return sol
    
    # Fallback: guaranteed valid representation
    return _fallback()


def _max_fill_sum(last, count):
    """
    Compute the maximum possible sum of `count` distinct unit fractions
    with denominators > last, as a fraction (num, den) in reduced form.
    Uses the count smallest available denominators: last+1, last+2, ..., last+count.
    """
    if count <= 0:
        return 0, 1
    num = 0
    den = 1
    for j in range(1, count + 1):
        d = last + j
        num = num * d + den
        den = den * d
    g = _gcd(num, den)
    return num // g, den // g


def _search_k(k, deadline):
    """
    Backtracking search for exactly k distinct denominators >= 5
    whose reciprocal sum is 1, using integer arithmetic with strong pruning.
    """
    chosen = []
    nodes_visited = [0]
    
    def rec(remaining, last, num, den):
        """
        Find `remaining` distinct denominators > last such that
        their reciprocal sum equals num/den (reduced fraction).
        """
        if time.time() > deadline:
            return False
        
        nodes_visited[0] += 1
        
        if remaining == 0:
            return num == 0
        
        if remaining == 1:
            if num == 0:
                return False
            if den % num == 0:
                d = den // num
                if d > last and d >= 5 and d <= _MAX_DEN:
                    chosen.append(d)
                    return True
            return False
        
        # remaining == 2: solve directly
        if remaining == 2:
            # Need 1/d1 + 1/d2 = num/den, d1 > last, d2 > d1
            # 1/d1 = num/den - 1/d2 => d2 = den / (num*d1 - den)
            # So num*d1 - den must divide den, and be positive
            # d1 > den/num (so that num/den - 1/d1 > 0)
            # d1 > last
            d1_low = max(last + 1, den // num + 1)
            if num <= 0:
                return False
            d1_high = min(_MAX_DEN, (2 * den) // num)  # since d2 > d1 >= d1, 1/d1 + 1/d2 < 2/d1 <= num/den => d1 <= 2*den/num
            if d1_low > d1_high:
                return False
            for d1 in range(d1_low, d1_high + 1):
                if time.time() > deadline:
                    return False
                t = num * d1 - den
                if t <= 0:
                    continue
                if den % t != 0:
                    continue
                d2 = den // t
                if d2 <= d1 or d2 > _MAX_DEN:
                    continue
                chosen.append(d1)
                chosen.append(d2)
                return True
            return False
        
        # remaining >= 3
        # Next denominator d must satisfy:
        #   d > last, d >= 5
        #   1/d < num/den  =>  d > den/num
        #   Sum of remaining terms from d onward must be >= num/den
        #   => remaining/d >= num/den  =>  d <= remaining*den/num
        
        d_low = max(last + 1, 5)
        if num > 0:
            d_low = max(d_low, den // num + 1)
        
        if num > 0:
            d_high = min(_MAX_DEN, (remaining * den) // num)
        else:
            d_high = _MAX_DEN
        
        if d_low > d_high:
            return False
        
        # Tighter pruning: compute exact max fill sum for (remaining-1) terms after d
        # For each candidate d, check if max fill >= new_num/new_den
        
        for d in range(d_low, d_high + 1):
            if time.time() > deadline:
                return False
            
            # 1/d must be < num/den, i.e., den < num*d
            if not (den < num * d):
                continue
            
            new_num = num * d - den
            new_den = den * d
            
            if new_num <= 0:
                continue
            
            g = _gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            # Pruning 1: quick bound - (remaining-1) terms each <= 1/(d+1)
            # Need (remaining-1)/(d+1) >= new_num/new_den
            # i.e., (remaining-1)*new_den >= new_num*(d+1)
            if (remaining - 1) * new_den < new_num * (d + 1):
                break
            
            # Pruning 2: tighter - compute exact max fill for (remaining-1) terms
            # after d: sum of 1/(d+1) + ... + 1/(d+remaining-1)
            # This is expensive, so only do it when the quick bound is close
            if (remaining - 1) * new_den < new_num * (d + 1) * 2:
                # Close to the bound, compute exact max fill
                max_num, max_den = _max_fill_sum(d, remaining - 1)
                if max_num * new_den < new_num * max_den:
                    break
            
            chosen.append(d)
            if rec(remaining - 1, d, new_num, new_den):
                return True
            chosen.pop()
        
        return False
    
    if rec(k, 4, 1, 1):
        return chosen[:]
    return None


def _fallback():
    """
    Guaranteed-valid baseline using subset DP over divisors of 2520.
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
