def construct():
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    
    # DP setup
    INF = 10**9
    dp = [INF] * (target + 1)
    parent = [0] * (target + 1) # 0 indicates not set
    dp[0] = 0
    
    # 0/1 Knapsack DP
    for d in divisors:
        # Iterate backwards to ensure each divisor is used at most once
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = d
                
    if dp[target] == INF:
        raise ValueError("No solution found")
        
    # Reconstruct
    chosen = []
    s = target
    while s > 0:
        d = parent[s]
        if d == 0:
            raise ValueError("Reconstruction failed")
        chosen.append(d)
        s -= d
        
    return chosen
