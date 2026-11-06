import time

def construct():
    deadline = time.time() + 85.0
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort(reverse=True)
    
    m = len(divisors)
    
    # Precompute prefix sums
    # prefix[i] = sum of divisors[0:i]
    prefix = [0] * (m + 1)
    for i in range(m):
        prefix[i + 1] = prefix[i] + divisors[i]
        
    # total sum of all divisors
    total_sum = prefix[m]

    def dfs(rem, k, start):
        """
        Find k distinct divisors from divisors[start:] that sum to rem.
        Returns a list of divisors or None.
        """
        if time.time() > deadline:
            return None
            
        if rem == 0:
            return [] if k == 0 else None
        if k == 0:
            return None
        if start >= m:
            return None
        if m - start < k:
            return None
            
        # Tight Min Sum Pruning
        # The k smallest elements in divisors[start:] are the global k smallest, 
        # i.e., divisors[m-k:m], provided m-k >= start.
        # Since m - start >= k, we have start <= m - k.
        min_sum = prefix[m] - prefix[m - k]
        if rem < min_sum:
            return None
            
        # Tight Max Sum Pruning
        # The k largest elements in divisors[start:] are divisors[start:start+k]
        max_sum = prefix[start + k] - prefix[start]
        if rem > max_sum:
            return None
            
        # Special case for k=1 to avoid deep recursion overhead
        if k == 1:
            # We need one divisor equal to rem from divisors[start:]
            # Since divisors are sorted descending, we can check if rem is in the subarray.
            # A simple loop is fine given m=60.
            for idx in range(start, m):
                if divisors[idx] == rem:
                    return [rem]
                if divisors[idx] < rem:
                    break
            return None

        for idx in range(start, m - k + 1):
            d = divisors[idx]
            if d > rem:
                continue
                
            next_rem = rem - d
            next_k = k - 1
            next_start = idx + 1
            
            # Pre-check bounds for the next recursive step to prune earlier
            if m - next_start < next_k:
                continue
                
            # Min sum for next_k items from next_start
            # The next_k smallest in divisors[next_start:] are divisors[m-next_k:m]
            # Valid if next_start <= m - next_k
            if next_start <= m - next_k:
                min_next = prefix[m] - prefix[m - next_k]
            else:
                # This case is handled by m - next_start < next_k check above
                continue
                
            if next_rem < min_next:
                continue
                
            # Max sum for next_k items from next_start
            max_next = prefix[next_start + next_k] - prefix[next_start]
            if next_rem > max_next:
                continue
                
            sub = dfs(next_rem, next_k, next_start)
            if sub is not None:
                return [d] + sub
                
        return None

    # Iterative deepening on k
    for k in range(1, m + 1):
        if time.time() > deadline:
            break
        result = dfs(target, k, 0)
        if result is not None:
            return result

    # Fallback: Greedy (should not be reached for this instance)
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    return chosen
