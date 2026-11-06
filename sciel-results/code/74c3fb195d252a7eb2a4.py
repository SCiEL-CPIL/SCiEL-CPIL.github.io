import time

def construct():
    """
    Finds the longest contiguous interval [start, start + length - 1] within [1, 1008]
    such that for all x, y in the interval, x * y % 1009 != 1.
    """
    p = 1009
    n = 1008
    
    # Precompute modular inverses
    # inv[x] is the unique y in [1, p-1] such that x*y % p == 1
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, p - 2, p)
        
    best_start = 1
    best_length = 0
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50s out of 90s allowed
    
    # Sliding window approach
    # L is the start of the current valid window
    L = 1
    for R in range(1, n + 1):
        # Check time limit
        if time.time() > deadline:
            break
            
        # Determine if R can be added to the window [L, R-1]
        # A window is invalid if it contains a pair (x, y) such that x*y % p == 1.
        # When adding R, new pairs are (R, y) for y in [L, R-1] and (R, R).
        
        if R * R % p == 1:
            # R is self-inverse. Any window containing R is invalid.
            # We must exclude R. Since R is the rightmost element, we set L to R+1.
            L = R + 1
        else:
            x_inv = inv[R]
            # If the inverse of R is in the current window [L, R-1], we must exclude it.
            # The inverse is unique.
            if x_inv >= L and x_inv < R:
                # We need to move L past x_inv
                L = x_inv + 1
        
        # Now the window [L, R] is valid (or empty if L > R)
        if L <= R:
            current_length = R - L + 1
            if current_length > best_length:
                best_length = current_length
                best_start = L

    # Fallback if no valid interval found (should not happen, e.g., [2,2] is valid)
    if best_length == 0:
        best_start = 2
        best_length = 1

    return [best_start, best_length]
