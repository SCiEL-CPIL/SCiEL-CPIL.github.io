import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin for 90s limit
    
    # 1. Generate divisors of 5040
    n = 5040
    divisors = [i for i in range(1, n + 1) if n % i == 0]
    target = 4979
    
    # 2. Try to find a subset of size k that sums to target
    # Minimum k is 1, maximum is len(divisors)
    # We iterate k from 1 upwards to ensure minimization
    
    # Optimization: If the sum of the k largest divisors is less than target, 
    # skip this k (though for small k this is unlikely to be the issue, 
    # but for large k it helps pruning if we were doing DP).
    # Here, simple combination check is fast enough for 60 items.
    
    # Sort divisors to allow potential pruning or just consistent ordering
    divisors.sort(reverse=True)
    
    # Quick check: Greedy solution
    # Greedy: 2520, 1680, 756, 21, 2 (sum=4979, count=5)
    # We check if k < 5 is possible.
    
    for k in range(1, len(divisors) + 1):
        # If even the sum of the k largest divisors is less than target, 
        # it's impossible for this k? No, smaller divisors sum to less, so yes.
        # If sum of k largest < target, no solution of size k exists.
        if sum(divisors[:k]) < target:
            continue
        
        # Check all combinations of size k
        # itertools.combinations is efficient enough for n=60, k<=5
        # For k > 5, the number of combinations grows, but we likely won't reach k > 5 
        # because we found a solution at k=5 via greedy logic (conceptually).
        # Actually, the loop will stop at the first valid k.
        
        for combo in combinations(divisors, k):
            if sum(combo) == target:
                # Time check before return
                if time.time() > deadline:
                    break
                return list(combo)
        
        if time.time() > deadline:
            break
            
    # If no solution found (should not happen), fallback to greedy
    # which we know works for this specific instance
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    return chosen

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin for 90s limit
    
    # 1. Generate divisors of 5040
    n = 5040
    divisors = [i for i in range(1, n + 1) if n % i == 0]
    target = 4979
    
    # DP approach to find minimum number of divisors summing to target
    # dp[s] = minimum number of divisors to sum to s
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # parent[s] = (prev_sum, divisor_used) to backtrack
    parent = [None] * (target + 1)
    
    # Iterate through each divisor
    for d in divisors:
        # Iterate sums downwards to ensure each divisor is used at most once (0/1 knapsack)
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)
        
        # Time check
        if time.time() > deadline:
             pass

    if dp[target] == INF:
        # Fallback, should not happen for this instance
        remainder = target
        chosen = []
        divisors.sort(reverse=True)
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    break
        return chosen

    # Backtrack to find the divisors
    result = []
    s = target
    while s > 0:
        if parent[s] is None:
            return []
        prev_s, d = parent[s]
        result.append(d)
        s = prev_s
        
    return result
# EVOLVE-BLOCK-END
