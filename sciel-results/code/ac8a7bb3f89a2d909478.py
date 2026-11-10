import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = 5040
    target = 4979
    
    # 1. Generate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    divisors.sort()
    N = len(divisors)
    
    # 2. DP Setup with NumPy
    # dp[i][s] = min number of divisors from divisors[0..i-1] to sum to s
    # We store the table to backtrack
    INF = 1000 # Max count is N=60, so 1000 is safe
    
    # dp shape: (N+1, target+1)
    dp = np.full((N + 1, target + 1), INF, dtype=np.int32)
    dp[0, 0] = 0
    
    for i in range(1, N + 1):
        d = divisors[i-1]
        prev_row = dp[i-1]
        curr_row = dp[i]
        
        # Copy prev_row to curr_row (case: don't pick d)
        # This is O(T)
        curr_row[:] = prev_row
        
        # Update with pick d
        # For s >= d: curr_row[s] = min(curr_row[s], prev_row[s-d] + 1)
        if d <= target:
            # prev_row[:-d] corresponds to indices 0..T-d
            # We want to update curr_row[d:]
            # candidate = prev_row[:-d] + 1
            # curr_row[d:] = np.minimum(curr_row[d:], candidate)
            
            # Note: prev_row[:-d] has length T+1-d
            # curr_row[d:] has length T+1-d
            candidates = prev_row[:-d] + 1
            # We need to ensure we don't overflow INF
            # np.minimum handles it
            
            # Only update where candidates are valid (prev_row[s-d] was not INF)
            # Actually min(INF, INF+1) is INF. min(val, INF) is val.
            # So direct min works.
            
            np.minimum(curr_row[d:], candidates, out=curr_row[d:])
            
        # Time check
        if time.time() > deadline:
             # If we break, we might not have full dp.
             # But N is small, this should not happen.
             pass

    # Check if solution exists
    if dp[N, target] >= INF:
        # Fallback
        return []

    min_count = dp[N, target]
    
    # 3. Backtrack
    chosen = []
    s = target
    for i in range(N, 0, -1):
        if time.time() > deadline:
            break
            
        d = divisors[i-1]
        
        # If dp[i][s] == dp[i-1][s], it means we didn't use d[i-1] OR we used it but it didn't reduce the count?
        # No, dp[i][s] is the min count using items 0..i-1.
        # If dp[i][s] < dp[i-1][s], we MUST have used d[i-1].
        # If dp[i][s] == dp[i-1][s], we COULD have used it, but it wasn't necessary for the count reduction.
        # However, if we use it, the remaining sum is s-d and count is dp[i-1][s-d].
        # We know dp[i][s] = min(dp[i-1][s], dp[i-1][s-d] + 1).
        
        if s >= d:
            # Check if using d[i-1] leads to the optimal count
            if dp[i][s] == dp[i-1][s-d] + 1:
                # It is possible that dp[i-1][s] is also equal to dp[i][s].
                # In that case, we have a choice.
                # If we pick d, we go to state (i-1, s-d).
                # If we don't pick d, we go to state (i-1, s).
                # Both are optimal in terms of count.
                # We can pick either.
                # Let's pick it if it's valid.
                chosen.append(d)
                s -= d
            else:
                # Must have come from dp[i-1][s]
                pass
        else:
            # s < d, cannot pick d
            pass
            
    return chosen
