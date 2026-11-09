"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for i in range(1, n + 1):
        if n % i == 0:
            divisors.append(i)
    divisors.sort(reverse=True)
    divisor_set = set(divisors)
    ndiv = len(divisors)
    
    # DP to find minimum number of divisors needed to sum to target
    # dp[s] = min count of divisors to sum to s
    INF = 1000
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    for d in divisors:
        # Update in reverse to ensure each divisor is used at most once
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                
    min_count = dp[target]
    
    if min_count == INF:
        # Fallback if no solution found (should not happen for this task)
        return []
        
    # Reconstruction: Find a subset of size min_count that sums to target
    # We use a recursive search with pruning
    # Sort divisors in descending order to find large divisors first (might help pruning)
    # Actually, we just need *any* valid subset.
    
    def find_subset(rem_sum, rem_count, start_idx):
        if rem_count == 0:
            if rem_sum == 0:
                return []
            else:
                return None
        if rem_sum < 0:
            return None
        # Pruning: if remaining sum is less than sum of smallest rem_count available divisors
        # Or if remaining sum is greater than sum of largest rem_count available divisors
        # Simple pruning: if rem_sum > sum(divisors[start_idx:start_idx+rem_count]): return None (if we take largest)
        # But we can pick any.
        
        # Check if it's possible to form rem_sum with rem_count divisors from divisors[start_idx:]
        # We can use a memoized DP or just iterate.
        # Given small size, simple iteration is fine.
        
        for i in range(start_idx, ndiv):
            d = divisors[i]
            if d > rem_sum:
                continue
            # Try picking d
            # We need rem_count - 1 more divisors from indices > i summing to rem_sum - d
            res = find_subset(rem_sum - d, rem_count - 1, i + 1)
            if res is not None:
                return [d] + res
        return None

    result = find_subset(target, min_count, 0)
    
    if result is None:
        # Fallback to brute force combinations if recursion failed (unlikely)
        for combo in combinations(divisors, min_count):
            if sum(combo) == target:
                return list(combo)
        return []
        
    return result
# EVOLVE-BLOCK-END
