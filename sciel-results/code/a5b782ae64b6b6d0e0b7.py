import time
import sys

def construct():
    n = 5040
    target = 4979
    
    # Generate all divisors of n
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors, maybe descending helps for early pruning in other methods, 
    # but for DP it doesn't strictly matter for correctness, 
    # though processing larger divisors first might fill the DP table faster? 
    # Actually, order doesn't matter for 0/1 knapsack DP correctness.
    divisors.sort()
    
    N = len(divisors)
    
    # dp[i][s] = minimum number of divisors from divisors[0..i-1] to sum to s
    # Initialize with a large number
    INF = 10**9
    # We can use a 2D list or a 1D list with backtracking info.
    # Given N=60 and Target=4979, a 2D list of size 61x4980 is ~300k integers.
    # Memory is fine.
    
    dp = [[INF] * (target + 1) for _ in range(N + 1)]
    dp[0][0] = 0
    
    for i in range(1, N + 1):
        d = divisors[i-1]
        for s in range(target + 1):
            # Option 1: Don't take divisor i
            val1 = dp[i-1][s]
            
            # Option 2: Take divisor i (if s >= d)
            val2 = INF
            if s >= d:
                if dp[i-1][s-d] != INF:
                    val2 = dp[i-1][s-d] + 1
            
            dp[i][s] = min(val1, val2)
            
    # Check if solution exists
    if dp[N][target] == INF:
        # Fallback: just return empty or raise, but problem guarantees solution
        return []

    # Backtrack to find the divisors
    selected = []
    s = target
    for i in range(N, 0, -1):
        d = divisors[i-1]
        # Check if we took this divisor
        # If dp[i][s] != dp[i-1][s], we must have taken it (unless both are INF, but we know it's not INF)
        # Actually, if dp[i][s] == dp[i-1][s] + 1 and s >= d, we took it.
        # More robust: 
        if s >= d and dp[i][s] == dp[i-1][s-d] + 1:
            selected.append(d)
            s -= d
        # Else, we didn't take it, so s remains same, and we move to i-1 implicitly by loop decrement
    
    # The selected list contains the divisors.
    # The problem asks for "at most 256 distinct positive divisors".
    # The count is len(selected).
    return selected

# EVOLVE-BLOCK-END
