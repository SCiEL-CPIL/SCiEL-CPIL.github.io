import math
import time

def construct():
    n = 5040
    target = 4979
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Generate divisors
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors in descending order to potentially find solutions with fewer larger divisors earlier?
    # DP doesn't care about order for optimality, but for reconstruction it doesn't matter.
    # However, sorting might help if we were doing greedy, but we are doing DP.
    divisors.sort()
    
    # DP setup
    # dp[s] = min number of divisors to sum to s
    INF = 10**9
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # parent[s] stores the last divisor added to reach sum s
    parent = [None] * (target + 1)
    
    # Fill DP table
    for d in divisors:
        if d > target:
            continue
        # Iterate backwards to ensure 0/1 property
        # Optimization: check time occasionally? No, loop is small.
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = d
                
    # Check if solution exists
    if dp[target] == INF:
        return []
        
    # Reconstruct the set of divisors
    chosen = []
    s = target
    while s > 0:
        d = parent[s]
        if d is None:
            break
        chosen.append(d)
        s -= d
        
    return chosen
