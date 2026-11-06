import time

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Get all divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    # Sort divisors descending to prioritize larger values (helps minimize count)
    divisors.sort(reverse=True)
    
    # Precompute suffix sums for pruning
    # suffix_sum[i] = sum of divisors[i:]
    suffix_sum = [0] * (len(divisors) + 1)
    for i in range(len(divisors) - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divisors[i]
        
    # Also precompute prefix sums? No, we need max sum of k items starting from idx.
    # max_sum(idx, k) = sum(divisors[idx : idx+k])
    
    def dfs(start_idx, remaining_sum, count_needed, current_path):
        if count_needed == 0:
            return current_path if remaining_sum == 0 else None
        
        if remaining_sum < 0:
            return None
            
        # Pruning: if not enough divisors left
        if start_idx + count_needed > len(divisors):
            return None
            
        # Pruning: if remaining_sum > sum of largest count_needed divisors available from start_idx
        # The largest count_needed divisors from start_idx are divisors[start_idx : start_idx + count_needed]
        max_possible = sum(divisors[start_idx : start_idx + count_needed])
        if remaining_sum > max_possible:
            return None
            
        # Pruning: if remaining_sum < sum of smallest count_needed divisors available from start_idx
        # The smallest count_needed divisors from start_idx are the last count_needed in the slice divisors[start_idx:]
        # i.e., divisors[len(divisors)-count_needed : len(divisors)] is not correct, 
        # it's divisors[start_idx + (len(divisors)-start_idx) - count_needed : len(divisors)]
        # Simpler: sum of the last 'count_needed' elements in divisors[start_idx:]
        min_possible = sum(divisors[len(divisors) - count_needed : len(divisors)]) # This is global min, not local.
        # Correct local min:
        local_slice = divisors[start_idx:]
        if len(local_slice) < count_needed:
            return None
        min_possible = sum(local_slice[-count_needed:])
        if remaining_sum < min_possible:
            return None
            
        # Check time
        if time.time() - start_time > DEADLINE:
            return None

        for i in range(start_idx, len(divisors)):
            d = divisors[i]
            if d > remaining_sum:
                # Since sorted descending, if d > remaining_sum, all subsequent are smaller? 
                # No, sorted descending means divisors[i] >= divisors[i+1].
                # So if divisors[i] > remaining_sum, we can't use it, but we can use smaller ones.
                # So we just skip.
                continue
            
            # Additional pruning: can we form the rest?
            # The rest must be formed by count_needed - 1 divisors from i+1 onwards.
            # Max sum of rest: sum(divisors[i+1 : i+1 + count_needed - 1])
            if count_needed - 1 > 0:
                max_rest = sum(divisors[i + 1 : i + 1 + count_needed - 1])
                if remaining_sum - d > max_rest:
                    continue
                min_rest = sum(divisors[len(divisors) - (count_needed - 1) : len(divisors)])
                # This min_rest is global, need local min from i+1
                local_rest_slice = divisors[i+1:]
                if len(local_rest_slice) < count_needed - 1:
                    continue
                min_rest = sum(local_rest_slice[-(count_needed - 1):])
                if remaining_sum - d < min_rest:
                    continue

            # Recurse
            result = dfs(i + 1, remaining_sum - d, count_needed - 1, current_path + [d])
            if result is not None:
                return result
                
        return None

    # Try increasing k to find the minimum number of divisors
    for k in range(1, len(divisors) + 1):
        # Optional: Prune k if sum of k largest < target
        if k > 0 and sum(divisors[:k]) < target:
            continue
            
        res = dfs(0, target, k, [])
        if res is not None:
            return res
            
    # Fallback: greedy (should not be reached for valid instances)
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
    return chosen
