import time

# Pre-compute the solution
# n = 5040
# target = 4979

def _precompute_solution():
    target = 4979
    n = 5040
    
    # Get divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    divisors.sort(reverse=True)
    N = len(divisors)
    
    INF = 10**9
    # dp[i][j] = min number of divisors from first i divisors to sum to j
    # Using 2D list for backtracking
    dp = [[INF] * (target + 1) for _ in range(N + 1)]
    choice = [[False] * (target + 1) for _ in range(N + 1)]
    dp[0][0] = 0
    
    for i in range(1, N + 1):
        d = divisors[i-1]
        # Optimization: Only iterate up to target
        for j in range(target + 1):
            # Option 1: Don't use d
            # dp[i][j] is already initialized to INF, but we need to copy from i-1
            # Actually, we can do this more efficiently by iterating j from 0 to target
            # and updating in place if we used 1D, but we need 2D for backtrack.
            
            # Copy from previous row
            # We can optimize by not copying if we just overwrite
            dp[i][j] = dp[i-1][j]
            choice[i][j] = False
            
            # Option 2: Use d
            if j >= d:
                if dp[i-1][j-d] + 1 < dp[i][j]:
                    dp[i][j] = dp[i-1][j-d] + 1
                    choice[i][j] = True

    if dp[N][target] == INF:
        return [] # Should not happen

    # Backtrack
    chosen = []
    i = N
    j = target
    while i > 0:
        if choice[i][j]:
            chosen.append(divisors[i-1])
            j -= divisors[i-1]
        i -= 1
        
    return chosen

# Compute once at module load
_SOLUTION = _precompute_solution()

def construct():
    # Return a copy to prevent external modification of the global state
    # though the task doesn't specify if the list is mutable/immutable
    return _SOLUTION.copy()
