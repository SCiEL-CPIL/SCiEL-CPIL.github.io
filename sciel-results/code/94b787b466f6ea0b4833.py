"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Get all divisors of 5040 that are <= target
    divisors = []
    for d in range(1, int(n**0.5) + 1):
        if n % d == 0:
            if d <= target:
                divisors.append(d)
            other = n // d
            if other != d and other <= target:
                divisors.append(other)
    
    # DP: dp[s] = minimum number of distinct divisors to achieve sum s
    INF = 10**9
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # parent tracking for reconstruction: parent[s] = (prev_sum, divisor_used)
    parent = [None] * (target + 1)
    
    # Sort divisors in descending order - may help find large-sum solutions earlier
    divisors.sort(reverse=True)
    
    for d in divisors:
        # Update dp in reverse to ensure 0/1 knapsack property (each divisor used once)
        # We only need to update sums s >= d
        for s in range(target, d - 1, -1):
            prev_val = dp[s - d]
            if prev_val + 1 < dp[s]:
                dp[s] = prev_val + 1
                parent[s] = (s - d, d)
        
        # Optional: check time, though this should be very fast (<0.1s)
        if time.time() - start_time > DEADLINE:
            break

    if dp[target] == INF:
        # Fallback: should not happen for this practical number instance
        return []

    # Reconstruct the solution
    chosen = []
    cur = target
    while cur > 0:
        prev_s, d = parent[cur]
        chosen.append(d)
        cur = prev_s
        
    return chosen
# EVOLVE-BLOCK-END
