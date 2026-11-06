import math
import time

def construct():
    start_time = time.time()
    n = 5040
    target = 4979

    # Precompute divisors of 5040
    divisors = [i for i in range(1, n + 1) if n % i == 0]
    # Sort descending to potentially prune earlier in DFS if we switched, but for DP order doesn't matter for correctness
    divisors.sort(reverse=True)
    
    num_divs = len(divisors)
    
    # DP: dp[s] = min number of divisors to sum to s
    # We use 1D DP for space efficiency, but need to reconstruct the set.
    # To reconstruct with 1D DP, we can store the "last item" used to achieve the optimal count for each sum.
    # However, 1D DP overwrites previous states.
    # Standard 0/1 knapsack reconstruction with 1D array requires careful backtracking or storing history.
    # Since target is ~5000 and num_divs is 60, 2D DP is very small (60 * 5000 = 300k ints).
    # Let's use 2D DP for simplicity and robustness of reconstruction.
    
    INF = float('inf')
    # dp[i][s] = min count using first i divisors (indices 0..i-1) to reach sum s
    dp = [[INF] * (target + 1) for _ in range(num_divs + 1)]
    dp[0][0] = 0
    
    for i in range(1, num_divs + 1):
        d = divisors[i-1]
        # Optimization: iterate s from 0 to target
        # If d > target, we can't take it, so dp[i][s] = dp[i-1][s] for all s
        if d > target:
            # Copy previous row
            dp[i] = dp[i-1][:]
        else:
            for s in range(target + 1):
                # Option 1: Don't take divisors[i-1]
                dp[i][s] = dp[i-1][s]
                # Option 2: Take divisors[i-1] if s >= d
                if s >= d:
                    val = dp[i-1][s-d] + 1
                    if val < dp[i][s]:
                        dp[i][s] = val
                    
    min_count = dp[num_divs][target]
    
    if min_count == INF:
        # Should not happen for this instance, but handle gracefully
        return []
        
    # Reconstruct
    result = []
    s = target
    for i in range(num_divs, 0, -1):
        if dp[i][s] != dp[i-1][s]:
            # We took divisors[i-1]
            result.append(divisors[i-1])
            s -= divisors[i-1]
            
    return sorted(result, reverse=True)
