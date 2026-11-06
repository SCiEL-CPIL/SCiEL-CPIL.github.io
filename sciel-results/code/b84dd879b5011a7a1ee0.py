import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin within 60s limit
    
    n = 5040
    target = 4979
    
    # Generate divisors of 5040
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    
    # Sort divisors in descending order to prioritize larger divisors
    # This can help find solutions with fewer terms earlier, though DP will find the global optimum.
    # More importantly, larger divisors first might lead to a "sparser" DP table initially,
    # but the main benefit is just deterministic ordering that might interact well with tie-breaking
    # if we were to implement one. For pure min-count DP, order doesn't change the final min count.
    divs.sort(reverse=True)
    
    # dp[s] = min number of divisors to sum to s
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # parent[s] = (prev_sum, divisor_used)
    parent = [None] * (target + 1)
    
    for d in divs:
        if d > target:
            continue
        
        # Check time periodically to avoid exceeding limit
        if time.time() > deadline:
            break
            
        # Update dp from target down to d
        # This ensures each divisor is used at most once
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)

    # Reconstruct solution
    if dp[target] == INF:
        return []

    current = target
    result = []
    while current > 0:
        prev, d = parent[current]
        result.append(d)
        current = prev
        
    return result
