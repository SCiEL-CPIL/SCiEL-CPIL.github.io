def construct():
    # Generate all divisors of 5040
    n = 5040
    target = 4979
    divisors = []
    for i in range(1, n + 1):
        if n % i == 0:
            divisors.append(i)
    
    # DP: dp[s] = minimum number of distinct divisors needed to sum to s
    # Use a large number as infinity
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # Also track which divisors were used for reconstruction
    # parent[s] = (previous_sum, divisor_used) to trace back
    parent = [None] * (target + 1)  # parent[s] = (s - d, d) that led to s
    
    for d in divisors:
        # Process in reverse to ensure each divisor used at most once
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)
    
    if dp[target] == INF:
        raise ValueError("No solution found")
    
    # Reconstruct the solution
    chosen = []
    s = target
    while s > 0:
        prev_s, d = parent[s]
        chosen.append(d)
        s = prev_s
    
    return chosen
