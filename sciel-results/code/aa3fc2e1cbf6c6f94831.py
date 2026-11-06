"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 85.0  # 5s safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Get all divisors of 5040 efficiently (O(sqrt(n)))
    divisors = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
        i += 1
    divisors.sort(reverse=True)  # largest first
    
    # Filter: only divisors <= target can be used
    divisors = [d for d in divisors if d <= target]
    
    # Quick check: is target itself a divisor? (k=1 solution)
    if target in divisors:
        return [target]
    
    # DP: dp[s] = minimum number of distinct divisors to achieve sum s
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    # parent tracking: for each sum, store which divisor was last added
    parent = [None] * (target + 1)  # parent[s] = (prev_sum, divisor)
    
    for d in divisors:
        if time.time() > DEADLINE:
            break
        # 0/1 knapsack: iterate backwards
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)
    
    if dp[target] == INF:
        # Fallback to greedy (should not happen)
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    break
        return chosen
    
    # Reconstruct the solution
    chosen = []
    s = target
    while s > 0:
        prev_s, d = parent[s]
        chosen.append(d)
        s = prev_s
    
    return chosen
# EVOLVE-BLOCK-END
