import time

def construct():
    n = 5040
    target = 4979
    
    # Calculate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort in descending order to find optimal (minimum count) solutions faster
    # This heuristic helps the DP explore larger items first, potentially finding
    # low-count solutions earlier in the state space, though for 0/1 knapsack
    # the order doesn't change the final optimal value, it can affect which
    # specific optimal solution is found if there are ties, or simply speed up
    # practical convergence if we were doing branch and bound. Here it's mostly
    # for consistency with the "greedy-like" intuition of using big numbers first.
    divisors.sort(reverse=True)
    
    # DP to find minimum number of divisors summing to target
    # dp[s] = minimum count of divisors to sum to s
    INF = float('inf')
    dp = [INF] * (target + 1)
    prev_sum = [-1] * (target + 1)
    prev_item = [0] * (target + 1)
    
    dp[0] = 0
    
    start_time = time.time()
    deadline = 50.0 # Safety margin before 60s limit
    
    # 0/1 Knapsack style update
    for d in divisors:
        # Check deadline
        if time.time() - start_time > deadline:
            break
            
        # 0/1 Knapsack style update
        # Iterate backwards to ensure each divisor is used at most once
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                prev_sum[s] = s - d
                prev_item[s] = d
                
    if dp[target] == INF:
        # Fallback: Greedy
        # This is highly unlikely to be needed as 5040 is a practical number
        # and the set of divisors is rich, but it's a safe exit.
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    break
        return chosen
        
    # Reconstruct the solution
    result = []
    s = target
    while s > 0:
        d = prev_item[s]
        if d == 0:
            break # Safety break
        result.append(d)
        s = prev_sum[s]
        
    return result
