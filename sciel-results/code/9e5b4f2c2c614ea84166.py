import time

def construct():
    # Constants
    N = 5040
    T = 4979
    MAX_TIME = 50.0
    
    start = time.time()
    
    # Generate divisors
    # Instead of simple range, we can use a slightly different method or just keep it simple but structured differently
    divs = []
    i = 1
    while i * i <= N:
        if N % i == 0:
            divs.append(i)
            if i != N // i:
                divs.append(N // i)
        i += 1
    
    # Sort descending for better pruning opportunities if we were using backtracking, 
    # but for DP, order matters for 0/1 constraint handling (descending sum index).
    # The order of processing items (divs) doesn't change the min count, 
    # but processing larger items first might allow earlier saturation of dp states if we were doing something else.
    # For standard DP, we just need to iterate all.
    divs.sort(reverse=True)
    
    num_items = len(divs)
    
    # DP Array: min_count_to_reach[sum]
    # Initialize with a large number
    INF = 10**9
    min_count = [INF] * (T + 1)
    min_count[0] = 0
    
    # To reconstruct the solution, we need to know which divisor was used to reach each sum.
    # parent[sum] = index of divisor in divs list
    parent_idx = [-1] * (T + 1)
    
    # Iterate through each divisor
    for idx in range(num_items):
        d = divs[idx]
        # Update dp in reverse order to ensure each divisor is used at most once
        # If we iterate forward, we would be allowing unlimited use (unbounded knapsack)
        # We only care about sums up to T
        for s in range(T, d - 1, -1):
            prev_count = min_count[s - d]
            if prev_count != INF:
                new_count = prev_count + 1
                if new_count < min_count[s]:
                    min_count[s] = new_count
                    parent_idx[s] = idx
        
        # Time check
        if time.time() - start > MAX_TIME:
            break
            
    if min_count[T] == INF:
        return []
        
    # Reconstruct the set of divisors
    result = []
    current_sum = T
    while current_sum > 0:
        p_idx = parent_idx[current_sum]
        if p_idx == -1:
            break
        d_val = divs[p_idx]
        result.append(d_val)
        current_sum -= d_val
        
    return sorted(result, reverse=True)
