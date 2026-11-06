import itertools

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort()
    
    # Precompute suffix sums for pruning?
    # Or just use the recursive search with basic bounds.
    # Since n is small, we can just try k=1..10
    
    # To speed up, we can check k=1,2,3,4 with combinations first
    # as they are fast.
    
    # Helper to check if a combination sums to target
    def check_combinations(k):
        for comb in itertools.combinations(divisors, k):
            if sum(comb) == target:
                return list(comb)
        return None

    # Try k=1 to 4 with itertools (fast enough)
    for k in range(1, 5):
        res = check_combinations(k)
        if res:
            return res
            
    # For k >= 5, use recursive search with pruning
    # We need to be careful with recursion depth and speed.
    # But k is likely small.
    
    # Let's implement a generic solver for k
    def solve_k(k):
        # Use recursion
        # To avoid deep recursion issues or overhead, we can use a stack or just recursion
        # Since k is small, recursion is fine.
        
        # Pruning helper: max sum of 'count' largest elements from divisors[start:]
        # min sum of 'count' smallest elements from divisors[start:]
        # We can precompute suffix sums of sorted divisors?
        # divisors are sorted ascending.
        # suffix_sum[i] = sum(divisors[i:])
        # But we need sum of 'count' largest from divisors[start:].
        # That is sum(divisors[-count:] if start is 0? No.)
        
        # Let's just use a simple recursive function with a global best or just return first found.
        
        def backtrack(start, k_remaining, target_remaining, path):
            if k_remaining == 0:
                if target_remaining == 0:
                    return path[:]
                return None
            if target_remaining < 0:
                return None
                
            # Pruning:
            # Max possible sum with k_remaining items from divisors[start:]
            # The largest k_remaining items in divisors[start:] are at the end of the list.
            # But we are restricted to indices >= start.
            # So max sum is sum(divisors[-k_remaining:]) if start == 0?
            # No, if start > 0, the available pool is divisors[start:].
            # The largest k_remaining in divisors[start:] are the last k_remaining elements of divisors[start:].
            
            # To make this efficient, we can precompute suffix sums of the sorted divisors.
            # suffix_sum[i] = sum(divisors[i:])
            # Then max sum of k items from divisors[start:] is NOT simply suffix_sum.
            # It is sum of the largest k items in the subarray divisors[start:].
            # Since divisors are sorted, the largest k items in divisors[start:] are:
            # divisors[-k:] if start == 0.
            # If start > 0, it's divisors[len(divisors)-k : len(divisors)]?
            # Wait, divisors[start:] is a suffix. The largest elements are at the end of the full list.
            # So the largest k elements in divisors[start:] are the last k elements of the full list,
            # UNLESS start is so large that there are fewer than k elements left?
            # No, we ensure k_remaining <= len(divisors) - start.
            
            # So max_sum = sum(divisors[-k_remaining:])
            # min_sum = sum(divisors[start:start+k_remaining])
            
            if target_remaining > sum(divisors[-k_remaining:]):
                return None
            if target_remaining < sum(divisors[start:start+k_remaining]):
                return None
                
            # Iterate
            # We can restrict the choice of the next divisor.
            # The next divisor d must be at least target_remaining - sum(largest k_remaining-1 from rest)
            # and at most target_remaining - sum(smallest k_remaining-1 from rest)
            
            # Simple iteration
            for i in range(start, len(divisors)):
                d = divisors[i]
                if d > target_remaining:
                    break # Since sorted, no further d will work
                # Optional: prune if d is too small
                # If d + sum(largest k_remaining-1 from i+1) < target_remaining, continue
                # If d + sum(smallest k_remaining-1 from i+1) > target_remaining, break? No, d is increasing, so sum increases.
                
                path.append(d)
                res = backtrack(i + 1, k_remaining - 1, target_remaining - d, path)
                if res:
                    return res
                path.pop()
            return None

        return backtrack(0, k, target, [])

    # Try k=5 to 10
    for k in range(5, 11):
        res = solve_k(k)
        if res:
            return res
            
    # Fallback to greedy if not found (should not happen)
    remainder = target
    chosen = []
    for d in sorted(divisors, reverse=True):
        if d <= remainder:
            chosen.append(d)
            remainder -= d
    return chosen
