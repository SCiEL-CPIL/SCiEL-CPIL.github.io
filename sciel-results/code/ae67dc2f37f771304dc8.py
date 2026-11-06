import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    n = 5040
    target = 4979
    
    # 1. Get all distinct divisors of n
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors ascending. 
    # Order doesn't matter for 0/1 DP correctness as long as we iterate sums in reverse.
    divisors.sort()
    N = len(divisors)
    
    # 2. Dynamic Programming
    # We want to minimize the number of divisors.
    # dp[s] = minimum number of divisors needed to sum to s
    # parent[s] = index of the divisor in 'divisors' that was last used to optimally reach sum s
    #            (specifically, the divisor that caused the update that set dp[s] to its current value)
    
    INF = float('inf')
    dp = [INF] * (target + 1)
    parent = [-1] * (target + 1)
    
    dp[0] = 0
    
    # Fill DP table
    # For each divisor, update dp in reverse order to ensure 0/1 property
    for i in range(N):
        d = divisors[i]
        # Iterate s from target down to d
        # This ensures we don't use the same divisor multiple times
        if time.time() > deadline:
            # If we're running out of time, we might not finish all divisors.
            # However, with N=60 and target=4979, this loop is ~300k ops, 
            # which is < 1ms in Python. This guard is just for safety.
            break
            
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = i
    
    # 3. Check if solution exists
    if dp[target] == INF:
        # Fallback: Greedy (should not happen for this instance)
        remainder = target
        chosen = []
        for d in sorted(divisors, reverse=True):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        if remainder == 0:
            return chosen
        else:
            return []
    
    # 4. Backtrack to find the divisors
    chosen = []
    s = target
    while s > 0:
        idx = parent[s]
        if idx == -1:
            # Should not happen if dp[target] is not INF
            break
        chosen.append(divisors[idx])
        s -= divisors[idx]
        
    # 5. Verification
    # Ensure all chosen numbers are distinct divisors of n and sum to target
    if len(chosen) != len(set(chosen)):
        # Duplicates found, which shouldn't happen with 0/1 DP, but safety check
        return []
        
    for d in chosen:
        if n % d != 0:
            return []
            
    if sum(chosen) != target:
        return []
        
    if len(chosen) > 256:
        return []
        
    return chosen
