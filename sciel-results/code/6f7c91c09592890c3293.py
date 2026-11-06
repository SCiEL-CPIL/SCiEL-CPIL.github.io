import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin within 60s limit
    
    n = 5040
    target = 4979
    
    # Generate all positive divisors of 5040
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    
    # Sort in descending order to try larger divisors first in heuristics
    divs.sort(reverse=True)
    m = len(divs)
    divisor_set = set(divs)
    
    # 1. Try k=1
    if target in divisor_set:
        return [target]
        
    # 2. Try k=2
    # d1 + d2 = target
    for d1 in divs:
        d2 = target - d1
        if d2 in divisor_set and d2 != d1:
            return [d1, d2]
            
    # 3. Try k=3
    # d1 + d2 + d3 = target
    # We iterate d1, d2. To avoid duplicates and ensure distinct, we can just check if (target - d1 - d2) is in set and distinct.
    # Since N is small (60), O(N^2) is fine.
    for i in range(m):
        d1 = divs[i]
        if d1 >= target:
            continue
        rem1 = target - d1
        for j in range(m):
            if i == j:
                continue
            d2 = divs[j]
            if d2 >= rem1:
                continue
            d3 = rem1 - d2
            if d3 in divisor_set and d3 != d1 and d3 != d2:
                return [d1, d2, d3]
                
    # 4. Fallback to DP for k >= 4
    # dp[s] = minimum number of divisors to achieve sum s
    INF = 10**9
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # parent[s] = (previous_sum, divisor_used) for reconstruction
    parent = [None] * (target + 1)
    
    # Iterate through divisors
    for idx, d in enumerate(divs):
        if d > target:
            continue
        
        # Time check
        if idx % 10 == 0 and time.time() > deadline:
            break
        
        # Update from high to low to ensure each divisor used at most once
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)
                
    # Reconstruct solution
    if dp[target] == INF:
        return []
    
    current = target
    result = []
    while current > 0:
        if parent[current] is None:
            return []
        prev, d_used = parent[current]
        result.append(d_used)
        current = prev
        
    return result
