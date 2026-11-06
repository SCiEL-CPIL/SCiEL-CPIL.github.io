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
        
    # suffix_min[k][i] = sum of the smallest k elements from divisors[i:]
    # For pruning: if we need k_rem elements from index 'start' onwards,
    # the minimum possible sum is the sum of the k_rem smallest elements in divisors[start:]
    # Since divisors is sorted descending, the smallest elements are at the end.
    # The k_rem smallest elements in divisors[start:] are divisors[D-k_rem:D] if D-k_rem >= start,
    # otherwise they are divisors[start:D] (but we only have D-start elements, and we need k_rem <= D-start)
    # Actually, the k_rem smallest elements in divisors[start:] are always the last k_rem elements of the array
    # as long as D - k_rem >= start. If D - k_rem < start, then all elements from start to D-1 are "small"
    # and we need exactly k_rem of them, which are divisors[D-k_rem:D] only if D-k_rem >= start.
    # Let me think again: divisors is sorted descending. divisors[start:] has D-start elements.
    # The smallest k_rem of these are the last k_rem elements of divisors[start:], which are divisors[D-k_rem:D]
    # provided D-k_rem >= start. If D-k_rem < start, then the smallest k_rem are divisors[start:D] but that's
    # D-start elements, and we need k_rem <= D-start, so the smallest k_rem are divisors[D-k_rem:D] only when
    # D-k_rem >= start. When D-k_rem < start, it means k_rem > D-start, which contradicts k_rem <= D-start.
    # So actually, as long as k_rem <= D - start, the k_rem smallest elements in divisors[start:] are 
    # divisors[D-k_rem:D].
    
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
        for i in range(D):
            d1 = divisors[i]
            if d1 >= target_val:
                continue
            d2 = target_val - d1
            if d2 in divisor_set and d2 != d1:
                return [d1, d2]
        return None

    # Specialized fast check for k=3
    def solve_k3(target_val):
        for i in range(D):
            d1 = divisors[i]
            if d1 >= target_val:
                continue
            rem1 = target_val - d1
            for j in range(i + 1, D):
                d2 = divisors[j]
                if d2 >= rem1:
                    continue
                rem2 = rem1 - d2
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
            
        # Precompute: for each start index and each k_rem, the min and max possible sums
        # min_sum[start][k_rem] = sum of k_rem smallest elements in divisors[start:]
        # max_sum[start][k_rem] = sum of k_rem largest elements in divisors[start:] = sum of divisors[start:start+k_rem]
        
        def backtrack(start, k_rem, target_rem, current_path):
            if k_rem == 0:
                if target_rem == 0:
                    return current_path[:]
                return None
            
            if len(divisors) - start < k_rem:
                return None
                
            # Pruning: Max sum - the k_rem largest elements from start onwards
            if start + k_rem > len(divisors):
                return None
            max_possible = range_sum(start, start + k_rem)
            if max_possible < target_rem:
                return None
                
            # Pruning: Min sum - the k_rem smallest elements from start onwards
            # These are divisors[D-k_rem:D] if D-k_rem >= start
            min_idx = D - k_rem
            if min_idx < start:
                # This shouldn't happen if k_rem <= D - start, but handle gracefully
                min_possible = range_sum(start, D)
            else:
                min_possible = range_sum(min_idx, D)
            if min_possible > target_rem:
                return None

            # Additional pruning: if target_rem is too large for the remaining elements
            # or too small
            
            for i in range(start, len(divisors)):
                d = divisors[i]
                if d > target_rem:
                    continue
                
                # Check if we have enough elements left
                if len(divisors) - (i + 1) < k_rem - 1:
                    break
                
                remaining = target_rem - d
                
                # Pruning: max sum of k_rem-1 elements from i+1 onwards
                if i + 1 + (k_rem - 1) <= len(divisors):
                    max_next = range_sum(i + 1, i + 1 + k_rem - 1)
                    if d + max_next < target_rem:
                        continue
                else:
                    break
                    
                # Pruning: min sum of k_rem-1 elements from i+1 onwards
                min_idx_next = D - (k_rem - 1)
                if min_idx_next < i + 1:
                    min_next = range_sum(i + 1, D)
                else:
                    min_next = range_sum(min_idx_next, D)
                if d + min_next > target_rem:
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
