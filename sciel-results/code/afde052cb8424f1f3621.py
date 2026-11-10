"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from functools import lru_cache

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 50  # 50s safety margin before 60s limit

    # Minimum possible k is 13 (harmonic sum 1/5+...+1/17 > 1, but 1/5+...+1/16 < 1)
    for k in range(13, 20):
        if time.time() > deadline:
            break
        result = _search_k(k, start, deadline)
        if result is not None:
            return result

    # Fallback: DP over divisors of a common denominator
    return _dp_fallback()


def _search_k(k, start, deadline):
    """DFS for exactly k distinct denominators >= 5 summing to 1."""
    # Use integer arithmetic: target = LCM of all denominators, but that's huge.
    # Instead, work with fractions as (num, den) pairs and use gcd reduction.
    
    def dfs(target_num, target_den, min_den, remaining, current):
        if time.time() > deadline:
            return None
        if remaining == 0:
            return current if target_num == 0 else None
        
        if target_num <= 0:
            return None
        
        # Max possible sum of 'remaining' terms from min_den:
        # 1/min_den + 1/(min_den+1) + ... + 1/(min_den+remaining-1)
        # Compare target_num/target_den with this sum
        # target > max_sum => prune
        # We check: target_num * (product of (min_den..min_den+remaining-1)) > target_den * sum(...)
        # Simpler: just compute the max sum as a fraction
        max_sum_num = 0
        max_sum_den = 1
        for i in range(remaining):
            d = min_den + i
            # Add 1/d to max_sum
            g = gcd(max_sum_den, d)
            new_num = max_sum_num * d // g + max_sum_den // g
            new_den = max_sum_den * d // g
            max_sum_num, max_sum_den = new_num, new_den
            # Reduce
            g = gcd(max_sum_num, max_sum_den)
            max_sum_num //= g
            max_sum_den //= g
        
        # Check if target > max_sum: target_num/target_den > max_sum_num/max_sum_den
        if target_num * max_sum_den > target_den * max_sum_num:
            return None
        
        # Determine range for next denominator d:
        # 1/d <= target => d >= target_den/target_num (ceil)
        # Also: target - 1/d >= 0 => d >= target_den/target_num
        # And: target - 1/d must be representable by (remaining-1) terms from d+1
        # Upper bound: 1/d >= target/remaining => d <= remaining*target_den/target_num
        # But also need d >= min_den
        
        min_d = max(min_den, (target_den + target_num - 1) // target_num)  # ceil(target_den/target_num)
        if min_d < 5:
            min_d = 5
        
        # Upper bound: d <= remaining * target_den / target_num
        max_d = min(10000000, (remaining * target_den) // target_num)
        if max_d < min_d:
            return None
        
        # Additional pruning: for the last few terms, be more specific
        if remaining == 1:
            # Need 1/d = target_num/target_den => d = target_den/target_num
            if target_den % target_num == 0:
                d = target_den // target_num
                if d >= min_d and d <= 10000000:
                    return current + [d]
            return None
        
        if remaining == 2:
            # Need 1/a + 1/b = target_num/target_den, a > b >= min_d
            # 1/a + 1/b = (a+b)/(ab) = target_num/target_den
            # target_den*(a+b) = target_num*a*b
            # target_num*a*b - target_den*a - target_den*b = 0
            # (target_num*a - target_den)(target_num*b - target_den) = target_den^2
            # Let X = target_num*a - target_den, Y = target_num*b - target_den
            # X*Y = target_den^2
            # a = (X + target_den)/target_num, b = (Y + target_den)/target_num
            # X and Y are divisors of target_den^2
            # a > b => X > Y (assuming target_num > 0)
            # Also a >= min_d+1, b >= min_d
            td2 = target_den * target_den
            # Find all divisors of td2
            divisors = []
            for i in range(1, int(td2**0.5) + 1):
                if td2 % i == 0:
                    divisors.append(i)
                    if i != td2 // i:
                        divisors.append(td2 // i)
            for X in divisors:
                Y = td2 // X
                if X >= Y:  # ensure a >= b
                    a = (X + target_den) / target_num
                    b = (Y + target_den) / target_num
                    if a != int(a) or b != int(b):
                        continue
                    a, b = int(a), int(b)
                    if a <= b:
                        continue
                    if b < min_d or a > 10000000:
                        continue
                    if b < 5 or a < 5:
                        continue
                    return current + [b, a]
            return None
        
        for d in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None
            # new_target = target - 1/d
            # = target_num/target_den - 1/d = (target_num*d - target_den) / (target_den*d)
            new_num = target_num * d - target_den
            if new_num <= 0:
                break
            new_den = target_den * d
            # Reduce
            g = gcd(new_num, new_den)
            new_num //= g
            new_den //= g
            
            # Check: can (remaining-1) terms from d+1 sum to new_num/new_den?
            # Max sum of (remaining-1) terms from d+1:
            ms_num = 0
            ms_den = 1
            for i in range(remaining - 1):
                dd = d + 1 + i
                gg = gcd(ms_den, dd)
                ms_num = ms_num * dd // gg + ms_den // gg
                ms_den = ms_den * dd // gg
                gg = gcd(ms_num, ms_den)
                ms_num //= gg
                ms_den //= gg
            
            if new_num * ms_den > new_den * ms_num:
                continue
            
            result = dfs(new_num, new_den, d + 1, remaining - 1, current + [d])
            if result is not None:
                return result
        return None
    
    return dfs(1, 1, 5, k, [])


def _dp_fallback():
    """Subset DP over divisors of 2520."""
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
