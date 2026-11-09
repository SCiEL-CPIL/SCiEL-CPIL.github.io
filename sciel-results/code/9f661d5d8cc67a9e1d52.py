import time

def construct():
    n = 5040
    target = 4979
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Generate divisors of 5040
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    divisors.sort(reverse=True)
    m = len(divisors)
    
    # Precompute prefix sums for quick range sum queries
    prefix_sum = [0] * (m + 1)
    for i in range(m):
        prefix_sum[i + 1] = prefix_sum[i] + divisors[i]
    
    def range_sum(i, j):
        if i >= j:
            return 0
        return prefix_sum[j] - prefix_sum[i]
    
    # Precompute min_sum[k] = sum of smallest k divisors = range_sum(m-k, m)
    # and max_sum_from[i][k] = sum of largest k from i onward = range_sum(i, i+k)
    # For efficiency, precompute max_possible[i][k] for all i, k where i+k <= m
    # But that's O(m^2) = 3600 entries, fine.
    
    def search(start, k, remaining, chosen):
        if time.time() > deadline:
            return None
        if k == 0:
            return chosen if remaining == 0 else None
        if start + k > m:
            return None
        
        # Max possible sum: take largest k from start
        max_possible = range_sum(start, start + k)
        if max_possible < remaining:
            return None
            
        # Min possible sum: take smallest k from available [start, m)
        # These are divisors[m-k:m], valid since m-k >= start (from start+k <= m)
        min_possible = range_sum(m - k, m)
        if min_possible > remaining:
            return None
            
        for i in range(start, m - k + 1):
            if time.time() > deadline:
                return None
            d = divisors[i]
            if d > remaining:
                continue
            
            # Prune: max sum of remaining k-1 from i+1
            if k - 1 == 0:
                max_rem = 0
            else:
                max_rem = range_sum(i + 1, i + 1 + k - 1)
            if max_rem < remaining - d:
                continue
            
            # Prune: min sum of remaining k-1 from i+1
            if k - 1 == 0:
                min_rem = 0
            else:
                # Smallest k-1 from [i+1, m) are divisors[m-(k-1):m]
                # Valid since m-(k-1) >= i+1 when i <= m-k
                min_rem = range_sum(m - (k - 1), m)
            if min_rem > remaining - d:
                # As i increases, d decreases, remaining-d increases, min_rem stays same
                # So this will remain true -> break
                break
                
            chosen.append(d)
            result = search(i + 1, k - 1, remaining - d, chosen)
            chosen.pop()
            if result is not None:
                return result
        
        return None
    
    for k in range(1, m + 1):
        if time.time() > deadline:
            break
        if k == 1:
            if target in divisors:
                return [target]
            continue
            
        chosen = []
        result = search(0, k, target, chosen)
        if result is not None:
            return result
            
    return []
