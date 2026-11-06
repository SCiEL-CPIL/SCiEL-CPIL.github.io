def construct():
    p = 1009
    n = 1008
    
    # Precompute modular inverses
    inv = [0] * p
    for x in range(1, p):
        inv[x] = pow(x, p-2, p)
        
    # Valid elements are in [2, 1007]
    # We use a sliding window (two-pointer) approach to find the longest
    # contiguous interval [left, right] such that for all x in the interval,
    # inv[x] is NOT in the interval.
    
    best_len = 0
    best_start = 2
    
    left = 2
    right = 1
    
    while left <= 1007:
        # Ensure right is at least left
        if right < left:
            right = left
            
        # Expand right as far as possible
        while right < 1007:
            next_val = right + 1
            # Conflict if inv[next_val] is in [left, right]
            if left <= inv[next_val] <= right:
                break
            right = next_val
            
        # [left, right] is now a maximal valid interval starting at left
        current_len = right - left + 1
        if current_len > best_len:
            best_len = current_len
            best_start = left
            
        left += 1
        
    return [best_start, best_len]
