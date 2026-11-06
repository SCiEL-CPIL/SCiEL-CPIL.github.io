"""Optimized: minimum-count subset of distinct divisors of 5040 summing to 4979."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def _all_divisors(n):
    """Generate all positive divisors of n, sorted ascending."""
    divs = []
    for d in range(1, int(n**0.5) + 1):
        if n % d == 0:
            divs.append(d)
            if d != n // d:
                divs.append(n // d)
    divs.sort()
    return divs

def construct():
    deadline = time.time() + 55.0  # safety margin before 60s limit
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040 (60 divisors)
    divisors = _all_divisors(n)
    m = len(divisors)
    
    # Sort divisors in descending order for better pruning in search
    divs_desc = sorted(divisors, reverse=True)
    
    # DP for minimum count subset sum
    # dp[s] = minimum number of divisors to achieve sum s
    INF = float('inf')
    dp = [INF] * (target + 1)
    parent = [None] * (target + 1)  # parent[s] = (divisor_used, prev_sum)
    dp[0] = 0
    
    # Process each divisor once (0/1 knapsack style)
    for d in divs_desc:
        # Process in reverse to ensure 0/1 constraint
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (d, s - d)
        
        # Check deadline
        if time.time() > deadline:
            break
    
    if dp[target] == INF:
        # Fallback: try BFS by count (should not happen for this instance)
        remainder = target
        chosen = []
        for d in range(n, 0, -1):
            if n % d == 0 and d <= remainder:
                chosen.append(d)
                remainder -= d
        assert remainder == 0, f"Greedy failed: remainder={remainder}"
        return chosen
    
    # Reconstruct solution
    chosen = []
    s = target
    while s > 0:
        d, prev_s = parent[s]
        chosen.append(d)
        s = prev_s
    
    # Verify
    assert len(chosen) <= 256, f"Too many divisors: {len(chosen)}"
    assert len(set(chosen)) == len(chosen), "Divisors must be distinct"
    assert sum(chosen) == target, f"Sum mismatch: {sum(chosen)} != {target}"
    for x in chosen:
        assert n % x == 0, f"{x} is not a divisor of {n}"
    
    return chosen
# EVOLVE-BLOCK-END
