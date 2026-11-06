import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    n = 5040
    target = 4979
    
    divisors = [i for i in range(1, n + 1) if n % i == 0]
    divisors.sort(reverse=True)
    
    num_divs = len(divisors)
    
    INF = float('inf')
    # dp[s] = min count to reach sum s
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # parent[s] = index of the divisor used to reach sum s optimally
    # We store the index in the divisors list (0-based)
    parent = [-1] * (target + 1)
    
    for i in range(num_divs):
        d = divisors[i]
        # Iterate sums in descending order to ensure 0/1 property
        # s from target down to d
        if time.time() > deadline:
            break
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = i

    min_count = dp[target]
    
    if min_count == INF:
        return []
        
    # Reconstruct
    result = []
    s = target
    while s > 0:
        idx = parent[s]
        if idx == -1:
            # Should not happen if dp[target] is not INF
            break
        result.append(divisors[idx])
        s -= divisors[idx]
        
    return sorted(result, reverse=True)
