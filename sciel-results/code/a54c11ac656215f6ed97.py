"""Shortest subset-sum via 0/1 knapsack DP for distinct divisors of 5040 summing to 4979."""
import time
import sys

N = 5040
TARGET = 4979
DEADLINE = 55.0


def _divisors_of(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)


def construct():
    start_time = time.time()
    
    divisors = _divisors_of(N)
    div_set = set(divisors)
    
    INF = float('inf')
    max_sum = TARGET
    dp = [INF] * (max_sum + 1)
    prev_sum = [-1] * (max_sum + 1)
    prev_div = [-1] * (max_sum + 1)
    dp[0] = 0
    
    # 0/1 knapsack: minimize count of distinct divisors to reach each sum
    for d in divisors:
        if time.time() - start_time > DEADLINE:
            break
        for s in range(max_sum, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                prev_sum[s] = s - d
                prev_div[s] = d
    
    if dp[TARGET] == INF:
        return []
    
    # Reconstruct
    chosen = []
    s = TARGET
    while s > 0:
        d = prev_div[s]
        if d < 0:
            return []
        chosen.append(d)
        s = prev_sum[s]
    
    # Verify
    if len(set(chosen)) != len(chosen):
        return []
    if not all(d in div_set for d in chosen):
        return []
    if sum(chosen) != TARGET:
        return []
    if len(chosen) > 256:
        return []
    
    return chosen
