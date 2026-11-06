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
    
    divisors.sort()
    num_divisors = len(divisors)
    
    # DP approach:
    # dp[s] = minimum number of divisors needed to sum to s
    # Also store parent pointers for reconstruction:
    # parent[s] = (prev_sum, divisor_index)
    
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # parent[s] will store a tuple (prev_sum, divisor)
    # We need to be careful to store enough info to reconstruct.
    # Since we process divisors one by one (0/1 knapsack), we can store:
    # used_divisor[s] = the divisor added to reach sum s in the optimal solution for the current subset of divisors
    # prev_sum[s] = the sum before adding that divisor
    
    # However, standard 0/1 knapsack DP updates in-place from high to low.
    # To reconstruct, we typically store `take[s]` boolean for each step, or store the previous state.
    # Given the small size, we can store `prev_sum[s]` and `divisor_used[s]` for the final optimal path.
    # But wait, the optimal path for sum `s` might change as we add more divisors.
    # A safer way: Store `dp` as a list of dictionaries or just re-run a DFS with the known optimal K.
    
    # Let's just find the minimum K first using DP.
    # Then use a fast DFS to find the specific combination for that K.
    # The DFS with pruning and known optimal K should be very fast because K is small.
    
    # Step 1: Find minimum K
    # We can optimize the DP to just track min count.
    for d in divisors:
        # Update from high to low to avoid using same divisor twice
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                
    min_k = dp[target]
    
    if min_k == INF:
        # Should not happen for this problem instance, but handle gracefully
        return []
        
    # Step 2: Reconstruct the solution using DFS with the known min_k
    # This is much faster than searching all k from 1 to N
    
    # Sort divisors descending for better pruning in DFS
    divs_desc = sorted(divisors, reverse=True)
    
    # Precompute suffix sums for pruning
    suffix_sums = [0] * (num_divisors + 1)
    for i in range(num_divisors - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + divs_desc[i]
        
    # Precompute prefix sums of the sorted list for max sum check?
    # Actually, for "max sum of k elements from idx", since list is sorted desc,
    # it's sum(divs_desc[idx : idx+k]).
    # We can precompute prefix sums of divs_desc to make this O(1).
    prefix_sums = [0] * (num_divisors + 1)
    for i in range(num_divisors):
        prefix_sums[i + 1] = prefix_sums[i] + divs_desc[i]
        
    def max_sum_from(idx, count):
        if count == 0:
            return 0
        if idx + count > num_divisors:
            return -1
        return prefix_sums[idx + count] - prefix_sums[idx]
        
    def dfs(idx, remaining, count, current_k, selected):
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
            
        remaining_slots = current_k - count
        if remaining_slots == 0:
            return None
            
        # Pruning: Can we reach 'remaining' with 'remaining_slots' items from idx onwards?
        max_possible = max_sum_from(idx, remaining_slots)
        if max_possible < remaining:
            return None
            
        # Pruning: Min sum check?
        # The min sum with 'remaining_slots' items from idx onwards is sum of the smallest 'remaining_slots' items in divs_desc[idx:].
        # These are the last 'remaining_slots' items in the list.
        # Indices: num_divisors - remaining_slots to num_divisors - 1.
        # But we must ensure these indices are >= idx.
        # If num_divisors - remaining_slots < idx, then we don't have enough small items?
        # No, we have items from idx to end. The smallest ones are at the end.
        # So the smallest sum is sum(divs_desc[num_divisors-remaining_slots : num_divisors]) 
        # IF num_divisors - remaining_slots >= idx.
        # If num_divisors - remaining_slots < idx, then all available items are "larger" than the global smallest.
        # In that case, the min sum is sum(divs_desc[idx : idx+remaining_slots])? No, that's max.
        # The min sum from idx with fixed count is sum of the smallest count elements in divs_desc[idx:].
        # Since divs_desc is sorted descending, the smallest elements are at the end.
        # So we want sum(divs_desc[num_divisors-remaining_slots : num_divisors]).
        # But we can only pick from idx onwards.
        # So the candidates are divs_desc[idx : num_divisors].
        # The smallest 'remaining_slots' of these are the last 'remaining_slots' of the slice.
        # So indices: num_divisors - remaining_slots to num_divisors - 1.
        # We need to check if these indices are valid (>= idx).
        # If num_divisors - remaining_slots < idx, then we must pick elements starting from idx.
        # In that case, the "smallest" available are divs_desc[idx], ..., but wait, they are the LARGEST available in that restricted set?
        # No, divs_desc is global descending.
        # If we are at idx, the available set is divs_desc[idx:].
        # The smallest elements in this set are at the end of the list.
        # So the min sum is always sum(divs_desc[num_divisors-remaining_slots : num_divisors])
        # PROVIDED that num_divisors - remaining_slots >= idx.
        # If num_divisors - remaining_slots < idx, it means we are forced to pick some larger elements?
        # No, if we need 'remaining_slots' elements from divs_desc[idx:], and the global smallest 'remaining_slots' start before idx,
        # then we can't use those. We have to use elements from idx onwards.
        # The smallest elements in divs_desc[idx:] are the last 'remaining_slots' elements of the whole list?
        # Yes, because the list is sorted. The tail of the list contains the smallest numbers.
        # As long as we have enough elements (num_divisors - idx >= remaining_slots),
        # the smallest 'remaining_slots' elements in the available set divs_desc[idx:] are exactly
        # divs_desc[num_divisors-remaining_slots : num_divisors].
        # Wait, this is only true if num_divisors - remaining_slots >= idx.
        # If num_divisors - remaining_slots < idx, then the "smallest" available elements are actually
        # divs_desc[idx : idx + remaining_slots]? No, those are the largest in the available set.
        # The smallest in divs_desc[idx:] are still the ones at the very end of the list.
        # So the min sum is sum(divs_desc[num_divisors-remaining_slots : num_divisors]).
        # Is it possible that num_divisors - remaining_slots < idx?
        # This implies remaining_slots > num_divisors - idx.
        # But we checked `if idx + count > num_divisors`? No, we have `remaining_slots = current_k - count`.
        # We need `num_divisors - idx >= remaining_slots`.
        # If not, we can't pick that many elements.
        # So add check: if num_divisors - idx < remaining_slots: return None.
        
        if num_divisors - idx < remaining_slots:
            return None
            
        min_possible = prefix_sums[num_divisors] - prefix_sums[num_divisors - remaining_slots]
        if min_possible > remaining:
            return None
            
        # Option 1: Pick divs_desc[idx]
        if divs_desc[idx] <= remaining:
            selected.append(divs_desc[idx])
            res = dfs(idx + 1, remaining - divs_desc[idx], count + 1, current_k, selected)
            if res is not None:
                return res
            selected.pop()
            
        # Option 2: Skip divs_desc[idx]
        res = dfs(idx + 1, remaining, count, current_k, selected)
        if res is not None:
            return res
            
        return None

    result = dfs(0, target, 0, min_k, [])
    if result is not None:
        return result
        
    # Fallback (should not be reached)
    return []
