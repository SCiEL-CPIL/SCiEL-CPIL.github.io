import time

def construct():
    start_time = time.time()
    deadline = 90  # seconds
    
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort in descending order to try large divisors first
    divisors.sort(reverse=True)
    num_divisors = len(divisors)
    
    # Precompute suffix sums for pruning
    suffix_sums = [0] * (num_divisors + 1)
    for i in range(num_divisors - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + divisors[i]
        
    # Precompute sums of largest k elements from index i
    # This is a bit expensive to precompute for all i, k.
    # Instead, we can just use the suffix sum for the "max possible sum" check.
    # For "min possible sum" with fixed count, we'd need the smallest ones, but we are picking from a sorted list.
    # Actually, for a fixed count k_remaining, the maximum sum we can get from indices >= idx is the sum of the first k_remaining elements in divisors[idx:].
    
    def max_sum_from(idx, count):
        if count == 0:
            return 0
        if idx + count > num_divisors:
            return -1 # Not enough elements
        # Sum of divisors[idx] to divisors[idx+count-1]
        # Since sorted descending, these are the largest available
        s = 0
        for j in range(idx, idx + count):
            s += divisors[j]
        return s

    def dfs(idx, remaining, count, current_k, selected):
        # Check deadline
        if time.time() > deadline - 5:
            return None
            
        if count == current_k:
            if remaining == 0:
                return list(selected)
            else:
                return None
                
        if idx == num_divisors:
            return None
            
        if remaining < 0:
            return None
            
        # Pruning 1: Can we reach 'remaining' with the remaining elements?
        # Max sum we can form with (current_k - count) elements from idx onwards
        remaining_slots = current_k - count
        if remaining_slots == 0:
            return None
            
        max_possible = max_sum_from(idx, remaining_slots)
        if max_possible < remaining:
            return None
            
        # Pruning 2: Min sum? 
        # The minimum sum with 'remaining_slots' elements from idx onwards is the sum of the smallest 'remaining_slots' elements in divisors[idx:].
        # But we are iterating in descending order. 
        # If the minimum possible sum (smallest elements) is greater than remaining, we can't reach it?
        # No, we want exactly 'remaining'. If the smallest sum we can form is > remaining, we are stuck.
        # Smallest 'remaining_slots' elements are at the end of the list.
        # But we are restricted to indices >= idx.
        # So the smallest elements available are divisors[num_divisors-1], ..., divisors[num_divisors-remaining_slots].
        # However, we must ensure these indices are >= idx.
        # If idx > num_divisors - remaining_slots, we don't have enough small elements? 
        # Actually, if we have 'remaining_slots' elements left to pick, and we are at index 'idx', 
        # the smallest sum we can make is sum(divisors[num_divisors-remaining_slots : num_divisors]) 
        # BUT only if those indices are >= idx. 
        # If idx is large, the available elements are divisors[idx...]. The smallest ones are at the end.
        
        # Let's just use the max_sum pruning for now, it's usually strong enough for this size.
        
        # Option 1: Pick divisors[idx]
        if divisors[idx] <= remaining:
            selected.append(divisors[idx])
            res = dfs(idx + 1, remaining - divisors[idx], count + 1, current_k, selected)
            if res is not None:
                return res
            selected.pop()
            
        # Option 2: Skip divisors[idx]
        res = dfs(idx + 1, remaining, count, current_k, selected)
        if res is not None:
            return res
            
        return None

    # Try increasing k
    for k in range(1, num_divisors + 1):
        # Optional: Quick check if max sum of k largest divisors is >= target
        if k <= num_divisors:
            max_sum_k = sum(divisors[:k])
            if max_sum_k < target:
                continue # Even the k largest can't reach target
        
        result = dfs(0, target, 0, k, [])
        if result is not None:
            return result
            
    # Fallback to greedy if search fails or times out (should not happen for k=5)
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    return chosen
