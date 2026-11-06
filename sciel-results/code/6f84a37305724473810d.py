from fractions import Fraction
import time
from math import gcd

_DEADLINE = 0

def construct():
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + 55
    
    # The greedy algorithm is fast and guaranteed to produce a valid solution.
    # The DFS is too slow and often fails to find a minimal solution within the time limit.
    # We return the greedy solution directly to ensure a valid, fast response.
    return greedy_solution()

def _find_k_terms(k):
    """Find k distinct denominators >= 5 summing to 1 using fast DFS."""
    global _DEADLINE
    
    def dfs(terms_rem, t_num, t_den, min_d, chosen):
        global _DEADLINE
        if time.time() > _DEADLINE:
            return None
        
        if terms_rem == 0:
            return chosen if t_num == 0 else None
        
        if terms_rem == 1:
            if t_num == 0:
                return None
            if t_den % t_num == 0:
                d = t_den // t_num
                if d >= min_d and d <= 10000000:
                    return chosen + [d]
            return None
        
        # Pruning: max sum with smallest denominators must be >= target
        # Use float for speed
        max_s = 0.0
        for i in range(terms_rem):
            max_s += 1.0 / (min_d + i)
        if max_s < t_num / t_den - 1e-15:
            return None
        
        # d bounds
        d_lo = max(min_d, -(-t_den // t_num))  # ceil(t_den/t_num)
        d_hi = min(10000000, (terms_rem * t_den) // t_num + 2)
        
        for d in range(d_lo, d_hi + 1):
            if time.time() > _DEADLINE:
                return None
            
            new_num = t_num * d - t_den
            if new_num <= 0:
                continue
            new_den = t_den * d
            
            # Check if remaining is achievable (float for speed)
            max_rest = 0.0
            for i in range(terms_rem - 1):
                max_rest += 1.0 / (d + 1 + i)
            if new_num / new_den > max_rest + 1e-15:
                break
            
            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            res = dfs(terms_rem - 1, new_num, new_den, d + 1, chosen + [d])
            if res is not None:
                return res
        
        return None
    
    return dfs(k, 1, 1, 5, [])

def greedy_solution():
    R = Fraction(1, 1)
    D = 5
    denoms = []
    while R > 0:
        inv_R = Fraction(R.denominator, R.numerator)
        d = max(D, inv_R.numerator // inv_R.denominator)
        if (inv_R.numerator % inv_R.denominator) != 0:
            d += 1
        if d < D:
            d = D
        denoms.append(d)
        R -= Fraction(1, d)
        D = d + 1
        if len(denoms) > 256:
            break
    return denoms
