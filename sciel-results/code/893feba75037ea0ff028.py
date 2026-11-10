"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 55  # 55s safety margin before 60s limit

    # Try k=8 first (most likely to find a solution quickly if one exists)
    result = _search_k(8, start, deadline)
    if result is not None:
        return result

    # Try k=9
    result = _search_k(9, start, deadline)
    if result is not None:
        return result

    # Try k=10
    result = _search_k(10, start, deadline)
    if result is not None:
        return result

    # Try k=11
    result = _search_k(11, start, deadline)
    if result is not None:
        return result

    # Try k=12
    result = _search_k(12, start, deadline)
    if result is not None:
        return result

    # Fallback: DP over divisors of 2520 (minimizing count)
    return _dp_fallback()


def _search_k(k, start, deadline):
    """DFS for exactly k distinct denominators >= 5 summing to 1 using exact Fraction arithmetic."""
    
    def dfs(target, min_den, remaining, current):
        if time.time() > deadline:
            return None
        if remaining == 0:
            return current if target == 0 else None
        
        if target <= 0:
            return None
        
        # Max sum of 'remaining' terms starting from min_den
        max_sum = Fraction(0)
        for i in range(remaining):
            max_sum += Fraction(1, min_den + i)
        if target > max_sum:
            return None
        
        # Upper bound on d: 1/d <= target => d >= 1/target
        # Also need: 1/d + max_sum(remaining-1, d+1) >= target
        # Simple bound: d <= remaining / target
        if target == 0:
            return None
        max_d = min(10000000, int(remaining / target) + 2)
        if max_d < min_den:
            return None
        
        for d in range(min_den, max_d + 1):
            if time.time() > deadline:
                return None
            
            # new_target = target - 1/d
            new_target = target - Fraction(1, d)
            if new_target < 0:
                break
            if new_target == 0:
                if remaining == 1:
                    return current + [d]
                else:
                    continue
            
            # Check max sum for remaining-1 terms from d+1
            max_rem = Fraction(0)
            for i in range(remaining - 1):
                max_rem += Fraction(1, d + 1 + i)
            if new_target > max_rem:
                continue
            
            result = dfs(new_target, d + 1, remaining - 1, current + [d])
            if result is not None:
                return result
        return None
    
    return dfs(Fraction(1, 1), 5, k, [])


def _dp_fallback():
    """Subset DP over divisors of 2520, minimizing count."""
    target = 2520
    # dp[sum] = (count, list)
    dp = {0: (0, [])}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        # Iterate over a copy of keys to avoid updating current iteration's items
        items = list(dp.items())
        for total, (count, values) in items:
            new_total = total + w
            if new_total <= target:
                new_count = count + 1
                if new_total not in dp or new_count < dp[new_total][0]:
                    dp[new_total] = (new_count, values + [d])
        if target in dp:
            return dp[target][1]
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
