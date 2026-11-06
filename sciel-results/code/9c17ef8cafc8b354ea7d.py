import time
import math

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # 1. Get divisors of 5040
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort(reverse=True)  # descending for greedy-like pruning
    
    N = len(divisors)
    
    # Quick check: k=1
    if target in divisors:
        return [target]
    
    # k=2: two-sum
    seen = set()
    for d in divisors:
        complement = target - d
        if complement in seen and complement != d:
            return [complement, d]
        seen.add(d)
    
    # k=3: three-sum with hashing
    for i in range(N):
        if time.time() > deadline - 5:
            break
        d1 = divisors[i]
        if d1 >= target:
            continue
        rem = target - d1
        # two-sum in remaining
        seen2 = set()
        for j in range(i + 1, N):
            d2 = divisors[j]
            if d2 >= rem:
                continue
            c2 = rem - d2
            if c2 in seen2 and c2 != d2:
                return [d1, c2, d2]
            seen2.add(d2)
    
    # k=4: four-sum with hashing (meet-in-the-middle style)
    if time.time() > deadline - 10:
        pass  # skip to DP
    
    # Use DP for exact minimum count
    INF = float('inf')
    # dp[s] = min number of divisors to achieve sum s
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # Store parent for backtracking: parent[s] = (prev_sum, divisor_used)
    parent = [None] * (target + 1)
    
    for i in range(N):
        d = divisors[i]
        # Iterate backwards to ensure each divisor used at most once
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)
        
        if time.time() > deadline - 5:
            break
    
    if dp[target] == INF:
        # Fallback: greedy
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    break
        return chosen
    
    # Backtrack to find the solution
    result = []
    s = target
    while s > 0 and parent[s] is not None:
        prev_s, d = parent[s]
        result.append(d)
        s = prev_s
    
    if sum(result) != target:
        # Fallback: greedy
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    break
        return chosen
    
    return result
