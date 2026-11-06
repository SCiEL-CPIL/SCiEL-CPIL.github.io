import math
import time

def construct():
    start_time = time.time()
    n = 5040
    target = 4979
    
    # Get divisors
    divisors = [i for i in range(1, n + 1) if n % i == 0]
    divisors.sort(reverse=True)
    
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # choice[s] stores the divisor used to reach sum s optimally
    # prev_sum[s] stores the previous sum before adding choice[s]
    choice = [0] * (target + 1)
    prev_sum = [0] * (target + 1)
    
    for d in divisors:
        # Iterate downwards to ensure 0/1 property (each divisor used at most once)
        # We only care about sums up to target
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                choice[s] = d
                prev_sum[s] = s - d
                
    # Check if solution exists
    if dp[target] == INF:
        return []
        
    # Reconstruct the set of divisors
    result = []
    s = target
    while s > 0:
        d = choice[s]
        if d == 0:
            # This should not happen if dp[target] is not INF, but safe guard
            break
        result.append(d)
        s = prev_sum[s]
        
    # The problem asks for distinct divisors. The DP ensures distinctness via 0/1 knapsack logic.
    # Sort result for consistent output (though not strictly required if just returning the list)
    # The task says "return at most 256 distinct positive divisors...".
    # It doesn't specify order, but sorted is nice.
    
    if time.time() - start_time > 55:
         pass

    return sorted(result, reverse=True)
