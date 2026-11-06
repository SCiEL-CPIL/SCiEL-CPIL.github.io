import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = [i for i in range(1, n + 1) if n % i == 0]
    divisors.sort(reverse=True)
    D = len(divisors)
    divisor_set = set(divisors)
    
    # Precompute prefix sums for pruning
    # prefix[i] = sum of divisors[0:i]
    prefix = [0] * (D + 1)
    for i in range(D):
        prefix[i + 1] = prefix[i] + divisors[i]
        
    def range_sum(lo, hi):
        """Sum of divisors[lo:hi]"""
        return prefix[hi] - prefix[lo]

    # Greedy fallback solution
    def greedy_solution():
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
            if remainder == 0:
                break
        if remainder == 0:
            return chosen
        return []

    # Specialized fast check for k=2
    def solve_k2(target_val):
        # d1 + d2 = target_val
        # Iterate d1, check if target_val - d1 is in set and > d1 (to avoid duplicates and ensure distinctness/order)
        # Since divisors are unique, we just need distinct indices.
        # We iterate d1 from largest to smallest.
        for i in range(D):
            d1 = divisors[i]
            if d1 >= target_val:
                continue
            d2 = target_val - d1
            if d2 < d1: # We want d2 < d1 to avoid checking pairs twice and ensure we don't use same index? 
                # Actually, if we iterate all i, we might find d1=3, d2=4976 (not div) or d1=4976, d2=3.
                # Since divisors are sorted desc, if d1 is large, d2 is small.
                # We just need to ensure d1 != d2.
                # If d2 is in divisor_set and d2 != d1, we have a candidate.
                # But we must ensure they are distinct divisors. Since divisor_set contains unique values, 
                # if d1 != d2, they are distinct.
                if d2 in divisor_set and d2 != d1:
                    return [d1, d2]
        return None

    # Specialized fast check for k=3
    def solve_k3(target_val):
        # d1 + d2 + d3 = target_val
        # Iterate d1, d2. Check if d3 = target_val - d1 - d2 is in set.
        # To avoid permutations, enforce d1 > d2 > d3.
        for i in range(D):
            d1 = divisors[i]
            if d1 >= target_val:
                continue
            rem1 = target_val - d1
            # d2 must be < d1
            for j in range(i + 1, D):
                d2 = divisors[j]
                if d2 >= rem1:
                    continue
                rem2 = rem1 - d2
                # d3 must be < d2
                # Check if rem2 is in divisor_set
                if rem2 in divisor_set and rem2 < d2:
                    return [d1, d2, rem2]
        return None

    # General backtracking for k >= 4
    def solve_k_general(k, target_val):
        if k == 0:
            return [] if target_val == 0 else None
        if k == 1:
            if target_val in divisor_set:
                return [target_val]
            return None
            
        def backtrack(start, k_rem, target_rem, current_path):
            if k_rem == 0:
                if target_rem == 0:
                    return current_path[:]
                return None
            
            if len(divisors) - start < k_rem:
                return None
                
            # Pruning: Max sum
            # The max sum of k_rem elements starting from 'start' is sum(divisors[start : start+k_rem])
            if start + k_rem > len(divisors):
                return None
            max_possible = range_sum(start, start + k_rem)
            if max_possible < target_rem:
                return None
                
            # Pruning: Min sum
            # The min sum of k_rem elements from the entire remaining pool (indices start to end)
            # is sum of the smallest k_rem available elements.
            # Since divisors are sorted desc, the smallest are at the end.
            # We need k_rem elements from indices [start, D-1].
            # The smallest k_rem elements in this range are divisors[D-k_rem : D] 
            # BUT only if D-k_rem >= start. If start is large, the smallest available are at the end.
            # Actually, the smallest k_rem elements available from index 'start' onwards are 
            # divisors[D-k_rem : D] provided D-k_rem >= start.
            # If D-k_rem < start, then the smallest k_rem are divisors[D-k_rem:start+k_rem]? No.
            # The available elements are divisors[start], divisors[start+1], ..., divisors[D-1].
            # The smallest k_rem of these are the last k_rem of this subarray: divisors[D-k_rem : D] 
            # if D-k_rem >= start.
            # If D-k_rem < start, then we don't have k_rem elements? No, we checked len - start >= k_rem.
            # So D - k_rem >= start is equivalent to D - start >= k_rem, which is true.
            # So the smallest k_rem available are always divisors[D-k_rem : D].
            # Wait, this is only true if we can pick any elements from the tail.
            # Yes, backtracking allows picking any element from index 'start' onwards.
            min_possible = range_sum(D - k_rem, D)
            if min_possible > target_rem:
                return None

            for i in range(start, len(divisors)):
                d = divisors[i]
                if d > target_rem:
                    continue
                
                # Prune: Can we form target_rem - d with k_rem - 1 elements from i+1 onwards?
                remaining = target_rem - d
                if k_rem - 1 == 0:
                    if remaining == 0:
                        return current_path + [d]
                    continue
                
                # Check if enough elements left
                if len(divisors) - (i + 1) < k_rem - 1:
                    break
                
                # Max sum of next k_rem-1 elements from i+1
                if i + 1 + (k_rem - 1) <= len(divisors):
                    max_next = range_sum(i + 1, i + 1 + k_rem - 1)
                    if d + max_next < target_rem:
                        continue
                else:
                    break
                    
                # Min sum of next k_rem-1 elements (smallest available)
                # Smallest k_rem-1 elements from i+1 onwards are divisors[D-(k_rem-1) : D]
                # Provided D-(k_rem-1) >= i+1.
                min_idx = D - (k_rem - 1)
                if min_idx < i + 1:
                    min_idx = i + 1
                # The smallest k_rem-1 elements in the range [i+1, D-1] are the last k_rem-1 elements of that range.
                # i.e., divisors[D-(k_rem-1) : D] if D-(k_rem-1) >= i+1.
                # Otherwise, it's just the whole range? No, we need exactly k_rem-1.
                # If the range length is L, the smallest k_rem-1 are the last k_rem-1 elements.
                min_next = range_sum(D - (k_rem - 1), D)
                # Wait, if start index i+1 is greater than D-(k_rem-1), the smallest elements are still at the end of the array.
                # So min_next is always range_sum(D - (k_rem - 1), D) as long as we have enough elements.
                if d + min_next > target_rem:
                    # Even taking the smallest possible remaining elements, the sum is too high.
                    # This means d is too small? No, d is fixed.
                    # If d + min_possible_remaining > target_rem, then this d is too small?
                    # No, if d is small, remaining is large.
                    # If d + min_remaining > target_rem, it means the minimum possible sum with d is already > target_rem.
                    # So d is too small? No, if d is small, sum is small.
                    # Let's re-evaluate.
                    # We want sum = target_rem.
                    # We pick d. Remaining sum needed = target_rem - d.
                    # Min sum of remaining k_rem-1 elements is min_next.
                    # If min_next > target_rem - d, then it's impossible.
                    # So if d + min_next > target_rem, then target_rem - d < min_next.
                    # This implies we can't form the remainder.
                    # As i increases, d decreases, so target_rem - d increases.
                    # So if this fails for current d, it might succeed for smaller d?
                    # Yes. So we should NOT break, just continue.
                    continue

                current_path.append(d)
                result = backtrack(i + 1, k_rem - 1, remaining, current_path)
                if result:
                    return result
                current_path.pop()
            
            return None
        
        return backtrack(0, k, target_val, [])

    # Try k from 1 upwards
    for k in range(1, D + 1):
        if time.time() > deadline:
            return greedy_solution()
            
        result = None
        if k == 1:
            if target in divisor_set:
                result = [target]
        elif k == 2:
            result = solve_k2(target)
        elif k == 3:
            result = solve_k3(target)
        else:
            result = solve_k_general(k, target)
            
        if result is not None:
            return result
            
    return greedy_solution()
