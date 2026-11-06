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
    divisors.sort()  # ascending order for DP
    
    m = len(divisors)
    
    # DP: dp[s] = min number of divisors to achieve sum s
    # Using 0/1 knapsack style: iterate divisors, update dp in reverse
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # Track parent for reconstruction
    parent = [-1] * (target + 1)  # which divisor was added
    prev_sum = [-1] * (target + 1)  # the sum before adding that divisor
    used_divisor = [-1] * (target + 1)  # index of the divisor used
    
    # Iterate through each divisor (ascending order)
    for idx, d in enumerate(divisors):
        if time.time() > deadline:
            break
        # Update dp in reverse to ensure 0/1 knapsack property (distinct divisors)
        # Only process sums up to target
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = d
                prev_sum[s] = s - d
                used_divisor[s] = idx
    
    if dp[target] != INF:
        # Backtrack to find the selected divisors
        selected = []
        s = target
        while s > 0:
            d = parent[s]
            if d == -1:
                # Should not happen if dp[target] is valid
                break
            selected.append(d)
            s = prev_sum[s]
        return selected
    
    # Fallback: iterative deepening DFS with pruning
    divisors_desc = sorted(divisors, reverse=True)
    m_desc = len(divisors_desc)
    
    # Precompute prefix sums for max possible sum with k items from start
    prefix = [0] * (m_desc + 1)
    for i in range(m_desc):
        prefix[i + 1] = prefix[i] + divisors_desc[i]
    
    # Precompute suffix sums for min possible sum check
    suffix_sum = [0] * (m_desc + 1)
    for i in range(m_desc - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divisors_desc[i]
    
    def dfs(rem, k, start):
        """Try to pick exactly k divisors from divisors_desc[start:] summing to rem."""
        if time.time() > deadline:
            return None
        if k == 0:
            return [] if rem == 0 else None
        if rem < 0:
            return None
        # Not enough elements left
        if m_desc - start < k:
            return None
        # Max possible sum with k items from divisors_desc[start:]
        if start + k > m_desc:
            return None
        max_possible = prefix[start + k] - prefix[start]
        if rem > max_possible:
            return None
        # Min possible sum: at least k * smallest divisor
        min_possible = k * divisors_desc[-1]
        if rem < min_possible:
            return None
        # Also check: can we reach rem with the remaining elements?
        if suffix_sum[start] < rem:
            return None
        
        for idx in range(start, m_desc - k + 1):
            d = divisors_desc[idx]
            if d > rem:
                continue
            # After picking d, need k-1 more from divisors_desc[idx+1:] summing to rem-d
            sub = dfs(rem - d, k - 1, idx + 1)
            if sub is not None:
                return [d] + sub
        return None
    
    # Try increasing depths
    for k in range(1, m_desc + 1):
        if time.time() > deadline:
            break
        result = dfs(target, k, 0)
        if result is not None:
            return result
    
    # Final fallback: greedy
    remainder = target
    chosen = []
    for d in divisors_desc:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    return chosen
