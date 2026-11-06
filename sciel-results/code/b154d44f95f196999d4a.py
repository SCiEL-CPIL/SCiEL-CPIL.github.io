def construct():
    n = 5040
    target = 4979
    
    # Generate all distinct positive divisors of n
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    
    # Filter divisors > target, as they cannot be part of the sum
    divisors = [d for d in divisors if d <= target]
    
    # DP to find minimum number of divisors to reach target sum
    # dp[s] stores the minimum number of divisors needed to sum to s
    # Using a large number to represent infinity
    INF = float('inf')
    max_sum = target
    dp = [INF] * (max_sum + 1)
    dp[0] = 0
    
    # To reconstruct the solution, we store the last item used to reach each sum
    # and the previous sum
    last_item = [None] * (max_sum + 1)
    prev_sum = [0] * (max_sum + 1)
    
    # Process each divisor
    for d in divisors:
        # Iterate downwards to ensure each divisor is used at most once (0/1 Knapsack)
        for s in range(max_sum, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                last_item[s] = d
                prev_sum[s] = s - d
                
    if dp[target] == INF:
        # Fallback, though theoretically should not happen for this instance
        # Greedy fallback
        remainder = target
        chosen = []
        for d in sorted(divisors, reverse=True):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    break
        return chosen

    # Backtrack to find the divisors
    result = []
    s = target
    while s > 0:
        d = last_item[s]
        if d is None:
            break # Should not happen if dp[target] != INF
        result.append(d)
        s = prev_sum[s]
        
    return result
