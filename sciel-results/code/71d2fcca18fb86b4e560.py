"""Rewritten program: exact minimum-cardinality subset-sum over divisors of 5040."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    target = 4979
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s kill

    # All distinct positive divisors of 5040
    n = 5040
    divisors = []
    for d in range(1, int(n**0.5) + 1):
        if n % d == 0:
            divisors.append(d)
            if d != n // d:
                divisors.append(n // d)
    
    # Filter out divisors larger than target (they can't be part of a valid sum)
    divisors = [d for d in divisors if d <= target]
    
    # Dynamic Programming to find minimum number of divisors summing to target
    # dp[s] stores the minimum number of divisors needed to sum to s
    # parent[s] stores (prev_sum, divisor_used) to reconstruct the solution
    
    INF = float('inf')
    dp = [INF] * (target + 1)
    parent = [None] * (target + 1)
    
    dp[0] = 0
    
    # Iterate through each divisor
    # Update dp from high to low to ensure each divisor is used at most once
    for d in divisors:
        if time.time() > deadline:
            break
        
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)
    
    # Reconstruct the solution
    if dp[target] == INF:
        return []  # No solution found (should not happen for practical numbers)
    
    solution = []
    current_sum = target
    while current_sum > 0:
        prev_sum, d = parent[current_sum]
        solution.append(d)
        current_sum = prev_sum
    
    return solution
# EVOLVE-BLOCK-END
