"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Compute all divisors of 5040
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    # Filter: only divisors <= target (can't use divisors larger than target)
    divisors = [d for d in divisors if d <= target]
    divisors.sort(reverse=True)
    m = len(divisors)
    
    # Quick check: is target itself a divisor?
    if target in divisors:
        return [target]
    
    # DP approach: 0/1 knapsack to find minimum number of divisors summing to target
    # dp[s] = minimum number of divisors needed to achieve sum s
    # parent[s] = (previous_sum, divisor_index) for reconstruction
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    parent = [None] * (target + 1)  # parent[s] = (prev_sum, divisor_index)
    
    for idx, d in enumerate(divisors):
        if d > target:
            continue
        # Process in reverse to avoid using same divisor twice (0/1 knapsack)
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, idx)
    
    if dp[target] == INF:
        # Fallback: greedy (should not happen for this instance, but safety net)
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
            if remainder == 0:
                break
        if remainder != 0:
            raise RuntimeError(f"Failed to find solution, remainder={remainder}")
        return chosen
    
    # Reconstruct the solution
    chosen = []
    s = target
    while s > 0:
        prev_s, idx = parent[s]
        chosen.append(divisors[idx])
        s = prev_s
    
    chosen.reverse()
    return chosen
# EVOLVE-BLOCK-END
