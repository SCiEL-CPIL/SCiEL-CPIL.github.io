"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Get all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort(reverse=True)
    
    m = len(divisors)
    
    # Dynamic programming for minimum cardinality subset sum
    # dp[s] = minimum number of divisors needed to sum to s, or INF if impossible
    # We track parent pointers to reconstruct the solution
    
    INF = float('inf')
    max_sum = target
    
    dp = [INF] * (max_sum + 1)
    dp[0] = 0
    
    # parent[s] = (prev_sum, divisor_index) to trace back
    prev_sum = [-1] * (max_sum + 1)
    prev_div_idx = [-1] * (max_sum + 1)
    
    for idx, d in enumerate(divisors):
        if time.time() > deadline:
            break
        # Process in reverse to avoid using the same divisor twice
        for s in range(max_sum, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                prev_sum[s] = s - d
                prev_div_idx[s] = idx
        # Check deadline after each divisor
        if time.time() > deadline:
            break
    
    if dp[target] != INF:
        # Reconstruct the solution
        chosen = []
        s = target
        while s > 0:
            idx = prev_div_idx[s]
            if idx < 0:
                break
            chosen.append(divisors[idx])
            s = prev_sum[s]
        
        # Verify the solution
        if len(set(chosen)) == len(chosen) and sum(chosen) == target and all(n % d == 0 for d in chosen):
            return chosen
    
    # Fallback: iterative deepening DFS with pruning
    # Precompute prefix sums for max possible sum with k items from index i onwards
    prefix = [0] * (m + 1)
    for i in range(m):
        prefix[i + 1] = prefix[i] + divisors[i]
    
    def dfs(remaining, start_idx, chosen, depth):
        if time.time() > deadline:
            return None
        if remaining == 0:
            return chosen[:]
        if len(chosen) >= depth:
            return None
        # Pruning: even taking all remaining items can't reach target
        if prefix[m] - prefix[start_idx] < remaining:
            return None
        # Pruning: can't fill remaining with (depth - len(chosen)) items
        slots = depth - len(chosen)
        if start_idx + slots > m:
            return None
        # Max sum with slots items from start_idx
        if prefix[min(start_idx + slots, m)] - prefix[start_idx] < remaining:
            return None
        
        for i in range(start_idx, m):
            d = divisors[i]
            if d > remaining:
                continue
            chosen.append(d)
            result = dfs(remaining - d, i + 1, chosen, depth)
            if result is not None:
                return result
            chosen.pop()
        return None
    
    for depth in range(1, 15):
        result = dfs(target, 0, [], depth)
        if result is not None:
            return result
    
    # Last resort: greedy
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
        if remainder == 0:
            break
    return chosen

# EVOLVE-BLOCK-END
