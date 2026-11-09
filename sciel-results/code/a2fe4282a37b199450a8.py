def construct():
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    
    INF = float('inf')
    # dp[s] stores the minimum number of divisors required to sum to s
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # prev[s] stores the divisor used in the optimal step to reach sum s
    prev = [None] * (target + 1)
    
    # 0/1 Knapsack DP
    # We iterate through each divisor and update the dp array from high to low sums
    for d in divisors:
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                prev[s] = d
    
    if dp[target] == INF:
        raise ValueError("No solution found")
    
    # Reconstruct the solution
    chosen = []
    s = target
    while s > 0:
        d = prev[s]
        if d is None:
            raise ValueError("Reconstruction failed")
        chosen.append(d)
        s -= d
        
    return chosen
