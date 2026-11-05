# EVOLVE-BLOCK-START
def construct():
    n = 5040
    target = 4979
    
    # Generate divisors of n
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Filter divisors <= target
    divisors = [d for d in divisors if d <= target]
    divisors.sort()
    
    num_divs = len(divisors)
    
    # DP table: dp[i][s] = min number of divisors from divs[0..i-1] to sum to s
    # Using a large number as infinity
    INF = 10**9
    dp = [[INF] * (target + 1) for _ in range(num_divs + 1)]
    dp[0][0] = 0
    
    # Fill DP table
    for i in range(1, num_divs + 1):
        d = divisors[i-1]
        for s in range(target + 1):
            # Option 1: Don't use divisor i-1
            dp[i][s] = dp[i-1][s]
            # Option 2: Use divisor i-1
            if s >= d:
                if dp[i-1][s-d] + 1 < dp[i][s]:
                    dp[i][s] = dp[i-1][s-d] + 1
    
    # Check if solution exists
    if dp[num_divs][target] == INF:
        # Fallback to greedy if no solution found (should not happen)
        remainder = target
        chosen = []
        for d in sorted(divisors, reverse=True):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        if remainder == 0:
            return chosen
        else:
            return [] # Should not happen

    # Backtrack to find the divisors
    chosen = []
    s = target
    for i in range(num_divs, 0, -1):
        d = divisors[i-1]
        # Check if divisor i-1 was used
        # If dp[i][s] == dp[i-1][s], it wasn't used
        # If dp[i][s] == dp[i-1][s-d] + 1, it was used
        if s >= d and dp[i][s] == dp[i-1][s-d] + 1:
            chosen.append(d)
            s -= d
    
    return chosen
# EVOLVE-BLOCK-END
