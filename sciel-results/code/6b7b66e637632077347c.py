import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort(reverse=True)  # Sort descending for better pruning
    
    # Filter divisors that are <= target
    divisors = [d for d in divisors if d <= target]
    m = len(divisors)
    
    # Precompute suffix sums for pruning
    # suffix_max[i] = sum of largest min(k, m-i) divisors from position i onwards
    # For pruning: if we need to pick k more divisors starting at index i,
    # the max sum is sum of divisors[i:i+k] (since sorted descending)
    # The min sum is sum of the k smallest from divisors[i:], which are divisors[m-k:] if m-k>=i
    
    # Iterative deepening DFS
    def dfs(k, idx, remaining, chosen):
        if remaining == 0:
            return list(chosen)
        if k == 0 or idx >= m or remaining < 0:
            return None
        if time.time() > deadline:
            return None
        
        # Pruning 1: min possible sum with k divisors from idx onwards
        # The k smallest divisors available from idx to m-1
        # Since sorted descending, they are at the end
        if k <= m - idx:
            min_possible = sum(divisors[m-k:]) if m-k >= idx else sum(divisors[idx:])
        else:
            min_possible = sum(divisors[idx:])
        if remaining < min_possible:
            return None
        
        # Pruning 2: max possible sum with k divisors from idx onwards
        # The k largest divisors from idx onwards are divisors[idx:idx+k]
        if k <= m - idx:
            max_possible = sum(divisors[idx:idx+k])
        else:
            max_possible = sum(divisors[idx:])
        if remaining > max_possible:
            return None
        
        for i in range(idx, m):
            d = divisors[i]
            if d > remaining:
                continue
            chosen.append(d)
            result = dfs(k-1, i+1, remaining - d, chosen)
            if result is not None:
                return result
            chosen.pop()
        return None
    
    # Try increasing k from 1 upwards
    for k in range(1, m+1):
        if time.time() > deadline:
            break
        result = dfs(k, 0, target, [])
        if result is not None:
            return result
    
    # Fallback: DP
    INF = float('inf')
    min_count = [INF] * (target + 1)
    min_count[0] = 0
    parent_div = [-1] * (target + 1)
    prev_sum = [-1] * (target + 1)
    
    for d in divisors:
        if time.time() > deadline:
            break
        for s in range(target, d - 1, -1):
            if min_count[s - d] != INF:
                new_count = min_count[s - d] + 1
                if new_count < min_count[s]:
                    min_count[s] = new_count
                    parent_div[s] = d
                    prev_sum[s] = s - d
    
    if min_count[target] == INF:
        return []
    
    result = []
    s = target
    while s > 0:
        d = parent_div[s]
        if d == -1:
            break
        result.append(d)
        s = prev_sum[s]
    return result
