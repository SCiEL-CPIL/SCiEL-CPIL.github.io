import time
import math

def construct():
    start_time = time.time()
    deadline = 90.0  # seconds
    
    n = 5040
    target = 4979
    
    # 1. Get divisors
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors. Order doesn't strictly matter for correctness of min count,
    # but consistent ordering helps debugging.
    divisors.sort()
    
    N = len(divisors)
    
    # 2. DP
    # dp[i][s] = min number of divisors from divisors[0..i-1] to sum to s
    # Initialize with a large number
    INF = float('inf')
    
    # To save memory and time, we can use 2 rows for dp values,
    # but we need the full history for backtracking.
    # Since N=60 and T=4979, storing full 2D table is fine.
    # dp[i][s]
    dp = [[INF] * (target + 1) for _ in range(N + 1)]
    dp[0][0] = 0
    
    # choice[i][s] = True if divisor i-1 was used to reach sum s
    # We can infer this from dp values during backtracking, so no need to store choice explicitly.
    # Check: if dp[i][s] == dp[i-1][s-d[i-1]] + 1, then it was used.
    # Note: It's possible that dp[i-1][s] == dp[i-1][s-d[i-1]] + 1 as well?
    # If dp[i-1][s] <= dp[i-1][s-d[i-1]] + 1, we don't use it.
    # If dp[i-1][s-d[i-1]] + 1 < dp[i-1][s], we use it.
    # If equal, we can choose either. To be deterministic, let's say we prefer NOT using it if equal?
    # Or prefer using it? It doesn't matter for the count, but might matter for the specific set.
    # The task just asks for *a* set with min count.
    
    for i in range(1, N + 1):
        d = divisors[i-1]
        for s in range(target + 1):
            # Option 1: Don't use d
            dp[i][s] = dp[i-1][s]
            
            # Option 2: Use d
            if s >= d:
                prev_val = dp[i-1][s-d]
                if prev_val + 1 < dp[i][s]:
                    dp[i][s] = prev_val + 1
            # Check deadline occasionally?
            # The loop is small, so probably not needed, but good practice.
            if time.time() > deadline - 1:
                # Fallback to greedy if we are running out of time (unlikely)
                return _greedy_fallback(n, target)

    # Check if solution exists
    if dp[N][target] == INF:
        # Should not happen for practical numbers, but just in case
        return _greedy_fallback(n, target)
        
    # 3. Backtrack
    result = []
    s = target
    for i in range(N, 0, -1):
        d = divisors[i-1]
        # Did we use d?
        # We used d if dp[i][s] == dp[i-1][s-d] + 1 AND (dp[i-1][s-d] + 1 < dp[i-1][s] OR we prefer this path)
        # Strictly:
        if s >= d and dp[i][s] == dp[i-1][s-d] + 1:
            # We need to be careful if dp[i-1][s] was also equal to this value.
            # If dp[i-1][s] == dp[i-1][s-d] + 1, then not using d is also optimal.
            # Let's just pick one. If dp[i][s] == dp[i-1][s], we didn't need to use d.
            # If dp[i][s] < dp[i-1][s], we MUST have used d.
            # If dp[i][s] == dp[i-1][s], we COULD have used d, but didn't need to.
            # To ensure we find *a* valid set, we should check:
            # If dp[i][s] == dp[i-1][s], we can skip d.
            # If dp[i][s] < dp[i-1][s], we must have taken d.
            
            if dp[i][s] < dp[i-1][s]:
                result.append(d)
                s -= d
            else:
                # dp[i][s] == dp[i-1][s]
                # We didn't strictly need to take d to achieve this min count at this step.
                # However, it's possible that taking d leads to a valid path and not taking it doesn't?
                # No, if dp[i-1][s] is the min count for sum s using first i-1 items,
                # and dp[i][s] == dp[i-1][s], then there exists a solution of that count using first i-1 items.
                # So we can safely ignore d and continue with i-1, s.
                pass
        # If s >= d and dp[i][s] == dp[i-1][s-d] + 1 but dp[i][s] == dp[i-1][s], we ignore d.
        # If s >= d and dp[i][s] < dp[i-1][s], we take d.
        
        # Wait, what if dp[i][s] == dp[i-1][s-d] + 1 AND dp[i][s] == dp[i-1][s]?
        # Then both paths are valid. Ignoring d is fine.
        
    # Verify
    if sum(result) != target:
         return _greedy_fallback(n, target)
         
    return result

def _greedy_fallback(n, target):
    # Same as seed program
    remainder = target
    chosen = []
    for d in range(n, 0, -1):
        if n % d == 0 and d <= remainder:
            chosen.append(d)
            remainder -= d
    return chosen
