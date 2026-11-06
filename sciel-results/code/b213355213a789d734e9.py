import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin, should be far below 60s limit

    n = 5040
    target = 4979

    # Generate divisors of 5040
    # 5040 = 2^4 * 3^2 * 5^1 * 7^1
    # We can just iterate 1..5040 for simplicity, it's fast enough.
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors. Order doesn't strictly matter for DP correctness,
    # but sorting can sometimes help with pruning if we were doing DFS.
    # For DP, it's fine. Let's sort descending to match previous style, 
    # though 1D DP iteration direction handles the 0/1 constraint.
    divisors.sort(reverse=True)
    
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    parent = [-1] * (target + 1)
    
    # 0/1 Knapsack DP
    # dp[s] = min number of divisors to sum to s
    # parent[s] = the divisor 'd' that was last added to achieve dp[s]
    
    for d in divisors:
        # Check time periodically? 
        # The loop is 60 * 5000 = 300k iterations. 
        # A time check inside the inner loop is expensive in Python.
        # We can check time in the outer loop.
        if time.time() > deadline:
            break
        
        # Iterate s from target down to d to ensure 0/1 property
        # (dp[s-d] is from the previous state, not updated by current d)
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = d

    min_count = dp[target]
    
    if min_count == INF:
        return []
        
    # Reconstruct the set of divisors
    result = []
    s = target
    while s > 0:
        d = parent[s]
        if d == -1:
            # This shouldn't happen if dp[target] is not INF
            # but break to avoid infinite loop
            break
        result.append(d)
        s -= d
        
    # The problem asks for distinct divisors. 
    # The DP ensures distinctness because each divisor is used at most once.
    # The result list contains the divisors.
    # Sort for consistent output (though order doesn't affect validity)
    return sorted(result, reverse=True)
