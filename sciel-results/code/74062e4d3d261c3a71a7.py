def construct():
    n = 5040
    target = 4979
    
    # Generate all distinct positive divisors of n
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    
    # Filter out divisors larger than the target
    divisors = [d for d in divisors if d <= target]
    divisors.sort()
    
    N = len(divisors)
    INF = float('inf')
    
    # dp[i][s] = min number of divisors from divisors[0..i-1] to sum to s
    dp = [[INF] * (target + 1) for _ in range(N + 1)]
    dp[0][0] = 0
    
    for i in range(1, N + 1):
        d = divisors[i-1]
        for s in range(target + 1):
            # Option 1: Don't take divisors[i-1]
            dp[i][s] = dp[i-1][s]
            # Option 2: Take divisors[i-1]
            if s >= d:
                val = dp[i-1][s - d] + 1
                if val < dp[i][s]:
                    dp[i][s] = val
                    
    if dp[N][target] == INF:
        return []
        
    # Backtrack to find the divisors
    result = []
    s = target
    i = N
    while i > 0 and s > 0:
        if dp[i][s] == dp[i-1][s]:
            i -= 1
        else:
            result.append(divisors[i-1])
            s -= divisors[i-1]
            i -= 1
            
    return result
