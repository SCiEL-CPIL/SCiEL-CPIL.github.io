import math
import time

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors of 5040
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Filter divisors <= target and sort
    divisors = [d for d in divisors if d <= target]
    divisors.sort()
    
    # Initialize DP arrays
    # dp[s] = min number of divisors to sum to s
    INF = 10**9
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # parent[s] = divisor that was added to reach s optimally
    parent = [0] * (target + 1)
    
    # Process each divisor
    for d in divisors:
        # For 0/1 knapsack, we must iterate backwards to avoid using d twice
        # s goes from target down to d
        # new_val = dp[s-d] + 1
        # If new_val < dp[s], update dp[s] and parent[s]
        
        # Optimization: Only consider s where dp[s-d] + 1 < dp[s]
        # We can break early if dp[s-d] is INF and dp[s] is already small? No, dp[s-d] might be reachable.
        
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = d
    
    # Check if solution exists
    if dp[target] >= INF:
        return []
        
    # Reconstruct
    chosen = []
    s = target
    while s > 0:
        d_val = parent[s]
        if d_val == 0:
            # Should not happen if dp[target] is valid
            break
        chosen.append(d_val)
        s -= d_val
        
    return chosen
