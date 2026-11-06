def construct():
    n = 5040
    target = 4979
    
    # Generate divisors efficiently
    divisors = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
        i += 1
    
    # Sort divisors (optional, but good for consistency)
    divisors.sort()
    
    INF = 10**9
    dp = [INF] * (target + 1)
    dp[0] = 0
    parent = [None] * (target + 1)
    
    for d in divisors:
        # Reverse loop for 0/1 knapsack
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)
                
    if dp[target] == INF:
        raise ValueError("No solution found")
        
    # Reconstruct
    chosen = []
    s = target
    while s > 0:
        prev_s, d = parent[s]
        chosen.append(d)
        s = prev_s
        
    return chosen
