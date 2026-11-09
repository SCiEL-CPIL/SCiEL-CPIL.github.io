def construct():
    MOD = 1009
    # The interval must be a subset of [2, 1007] because 1 and 1008 are self-inverse.
    # We use a sliding window approach to find the longest valid interval.
    # A window [left, right] is valid if it does not contain both x and inv(x) for any x.
    
    best_len = 0
    best_start = 2
    left = 2
    
    # Iterate through possible right endpoints
    for right in range(2, 1008):
        # Compute modular inverse of right modulo 1009
        inv_r = pow(right, -1, MOD)
        
        # If the inverse is already in the window [left, right-1],
        # we have a conflict. We must move 'left' to just after inv_r.
        # This ensures inv_r is excluded from the new window [left, right].
        if left <= inv_r < right:
            left = inv_r + 1
            
        # Calculate the length of the current valid window [left, right]
        current_len = right - left + 1
        
        # Update the best result
        if current_len > best_len:
            best_len = current_len
            best_start = left
            
    return [best_start, best_len]
