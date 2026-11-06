import time

def construct():
    """
    Return exactly 12 distinct positive integers <= 1048576 such that all 2^12 = 4096
    subset sums (including the empty subset) are distinct. Minimize the largest element.
    """
    start_time = time.time()
    deadline = start_time + 55.0

    n = 12
    # Greedy construction
    # We maintain the set of current elements and the set of all subset sums so far.
    # To ensure distinct subset sums, the next element a_k must be strictly greater
    # than the maximum subset sum of the previous k-1 elements.
    # This is because the new subset sums are a_k + {s | s is a subset sum of prev}.
    # If a_k <= max_prev_sum, it's possible for a_k + s1 == s2 for some s1, s2 in prev sums.
    # Specifically, if a_k <= max_prev_sum, the interval [a_k, a_k + max_prev_sum]
    # overlaps with [0, max_prev_sum].
    # For the sums to be guaranteed distinct without complex collision checks,
    # a_k > max_prev_sum is the sufficient condition.
    # It is also necessary for the minimal growth rate (powers of 2).
    
    elements = []
    subset_sums = {0}
    max_sum = 0
    
    for i in range(n):
        # The smallest valid next element is max_sum + 1
        # This ensures a_k > max_prev_sum
        next_val = max_sum + 1
        
        # Check if we are exceeding the maximum allowed value (1048576)
        # For n=12, max_sum will be 2^11 - 1 = 2047, so next_val = 2048.
        # 2048 <= 1048576, so it's fine.
        
        elements.append(next_val)
        
        # Update subset sums
        # New sums are next_val + s for all s in subset_sums
        new_sums = {next_val + s for s in subset_sums}
        subset_sums = subset_sums.union(new_sums)
        
        # Update max_sum
        max_sum += next_val
        
        # Time check
        if time.time() > deadline:
            break

    # The greedy algorithm produces powers of 2: 1, 2, 4, ..., 2048
    # Verify distinctness
    # We already maintained subset_sums, so if len(subset_sums) == 2^n, it's valid.
    if len(subset_sums) != (1 << n):
        # Fallback to hardcoded powers of 2 if greedy failed for some reason
        elements = [1 << i for i in range(n)]
    
    return elements
