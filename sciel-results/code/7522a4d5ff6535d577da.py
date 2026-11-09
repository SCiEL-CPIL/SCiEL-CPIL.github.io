import time
import sys

def construct():
    n = 5040
    target = 4979
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # 1. Get all divisors of n
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors. Order doesn't affect optimal count, but consistent order helps determinism.
    divisors.sort(reverse=True)
    
    N = len(divisors)
    
    # DP arrays
    # dp[s] stores the minimum number of divisors to sum to s
    # mask[s] stores the bitmask of divisors used
    INF = float('inf')
    dp = [INF] * (target + 1)
    mask = [0] * (target + 1)
    
    dp[0] = 0
    mask[0] = 0
    
    # 0/1 Knapsack DP
    # We iterate through each divisor
    for i in range(N):
        d = divisors[i]
        bit = 1 << i
        
        # Check time periodically to avoid timeout in worst-case scenarios (though unlikely here)
        if i % 10 == 0 and time.time() > deadline:
            break
            
        # We must iterate s backwards to ensure each divisor is used at most once
        # s goes from target down to d
        # Local variable binding for speed
        dp_local = dp
        mask_local = mask
        INF_local = INF
        
        for s in range(target, d - 1, -1):
            prev_count = dp_local[s - d]
            if prev_count != INF_local:
                new_count = prev_count + 1
                if new_count < dp_local[s]:
                    dp_local[s] = new_count
                    mask_local[s] = mask_local[s - d] | bit

    # Check if target is reachable
    if dp[target] == INF:
        # Should not happen for practical numbers, but handle gracefully
        return []
        
    # Reconstruct the set of divisors
    result_mask = mask[target]
    selected_divisors = []
    for i in range(N):
        if result_mask & (1 << i):
            selected_divisors.append(divisors[i])
            
    return selected_divisors
