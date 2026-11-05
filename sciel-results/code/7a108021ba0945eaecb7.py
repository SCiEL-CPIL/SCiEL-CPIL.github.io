def construct():
    n = 5040
    target = 4979
    
    # 1. Get divisors
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    divisors.sort(reverse=True)
    m = len(divisors)
    
    # 2. Precompute prefix sums for max sum calculation
    # prefix[i] = sum(divisors[0:i])
    prefix = [0] * (m + 1)
    for i in range(m):
        prefix[i + 1] = prefix[i] + divisors[i]
        
    # 3. Precompute min sums for each possible slot count
    # min_sum_for_slots[k] = sum of the k smallest divisors
    # Since divisors are sorted descending, the k smallest are divisors[m-k:m]
    min_sum_for_slots = [0] * (m + 1)
    # Note: min_sum_for_slots[k] is sum of last k elements
    # We can compute this efficiently or just sum slices for k in 1..m
    # m is 60, so summing slices is fine.
    for k in range(1, m + 1):
        min_sum_for_slots[k] = sum(divisors[m - k:m])
        
    # 4. Search for minimum k
    for k in range(1, m + 1):
        found = None
        
        def dfs(start_idx, remaining, current):
            nonlocal found
            if found is not None:
                return
            if len(current) == k:
                if remaining == 0:
                    found = current[:]
                return
            if remaining < 0:
                return
            
            slots = k - len(current)
            # Check if enough divisors left
            if slots > m - start_idx:
                return
                
            # Max possible sum with 'slots' divisors starting from start_idx
            # Sum of divisors[start_idx : start_idx + slots]
            max_sum = prefix[start_idx + slots] - prefix[start_idx]
            if max_sum < remaining:
                return
            
            # Min possible sum with 'slots' divisors
            # As derived, this is the sum of the globally smallest 'slots' divisors
            # because start_idx + slots <= m implies the smallest 'slots' are available
            min_sum = min_sum_for_slots[slots]
            if min_sum > remaining:
                return
            
            # Pruning: if remaining is smaller than the smallest available divisor
            # The smallest available divisor is divisors[m-1] (global smallest)
            # But specifically, the smallest divisor we can pick next is divisors[m-1]?
            # No, we are iterating i from start_idx.
            # The smallest divisor in the loop range is divisors[m-1].
            # If remaining < divisors[m-1], we can't pick anything.
            # This is covered by min_sum check?
            # If slots=1, min_sum is smallest divisor.
            # If remaining < min_sum, we return.
            # So it is covered.
            
            for i in range(start_idx, m):
                d = divisors[i]
                if d > remaining:
                    continue
                current.append(d)
                dfs(i + 1, remaining - d, current)
                current.pop()
                if found is not None:
                    return
        
        dfs(0, target, [])
        if found is not None:
            return found
            
    # Fallback (should not happen for this instance)
    return []
