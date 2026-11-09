import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s safety margin out of 90s
    
    n = 5040
    target = 4979
    
    # Generate all divisors of n
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors in ascending order for DP
    divisors.sort()
    
    # dp[s] will store the minimum number of divisors to sum to s,
    # and we also need to track which divisors were used to reconstruct the set.
    # Since we want to minimize the number of terms, we can use BFS-like DP or standard DP.
    # Let dp[s] be the minimum number of divisors to reach sum s.
    # Let parent[s] be the last divisor added to reach s.
    # Let prev[s] be the sum before adding the last divisor.
    
    # Initialize
    INF = float('inf')
    dp = [INF] * (target + 1)
    parent = [-1] * (target + 1) # stores index of divisor used
    prev_sum = [-1] * (target + 1) # stores the sum before the last step
    
    dp[0] = 0
    
    # Iterate through divisors
    for i, d in enumerate(divisors):
        # We need to be careful to use each divisor at most once.
        # So we iterate s from target down to d.
        # If we iterate upwards, we might use the same divisor multiple times.
        # So: for s in range(target, d-1, -1)
        
        if time.time() > deadline:
            break
            
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = i
                prev_sum[s] = s - d
    
    # Reconstruct the solution
    if dp[target] == INF:
        # Fallback (should not happen for practical numbers)
        # Try a simple greedy as last resort
        remaining = target
        chosen = []
        for d in reversed(divisors):
            if d <= remaining:
                chosen.append(d)
                remaining -= d
                if remaining == 0:
                    return chosen
        return chosen
        
    # Backtrack
    current_sum = target
    chosen = []
    while current_sum > 0:
        idx = parent[current_sum]
        chosen.append(divisors[idx])
        current_sum = prev_sum[current_sum]
        
    return chosen
