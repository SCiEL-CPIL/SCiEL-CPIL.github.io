def construct():
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    
    # DP: dp[s] = min count to reach sum s
    # Also track predecessor for reconstruction
    INF = float('inf')
    dp = [INF] * (target + 1)
    prev = [None] * (target + 1)
    dp[0] = 0
    
    for d in divisors:
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                prev[s] = (s - d, d)
                
    if dp[target] == INF:
        return [] # Should not happen
        
    # Reconstruct
    res = []
    s = target
    while s > 0:
        ps, d = prev[s]
        res.append(d)
        s = ps
        
    return res
