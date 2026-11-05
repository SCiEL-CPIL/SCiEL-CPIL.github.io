import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    MOD = 1009
    N = 1008

    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    # Self-inverse elements (x^2 == 1 mod 1009): {1, 1008}
    is_self_inv = [False] * (MOD + 1)
    for x in range(1, MOD):
        if x * x % MOD == 1:
            is_self_inv[x] = True

    # Two-pointer sliding window
    # L is the start of the current valid window
    # We iterate R from 1 to N
    # in_window[x] is True if x is in [L, R]
    
    in_window = [False] * (MOD + 1)
    best_len = 0
    best_start = 1
    L = 1
    
    for R in range(1, N + 1):
        # Check if R itself is a self-inverse
        if is_self_inv[R]:
            # R cannot be in any valid window
            # The window must end before R or start after R
            # Since we are expanding R, we must move L to R+1
            # And clear the window
            # Actually, if R is self-inverse, no window containing R is valid.
            # So the current window [L, R-1] was the last valid one ending at R-1.
            # We record its length before resetting.
            # The window [L, R-1] is valid.
            curr_len = R - L
            if curr_len > best_len:
                best_len = curr_len
                best_start = L
            
            # Reset for next R
            # All elements in [L, R-1] are no longer "active" in the sense that 
            # we are starting a fresh window after R.
            # We need to clear in_window for [L, R-1].
            for x in range(L, R):
                in_window[x] = False
            L = R + 1
            continue
        
        # Check if inv[R] is already in the window [L, R-1]
        partner = inv[R]
        if in_window[partner]:
            # Conflict! We must move L past 'partner'
            # The new L will be partner + 1
            # We need to remove elements from old L to partner from in_window
            new_L = partner + 1
            for x in range(L, new_L):
                in_window[x] = False
            L = new_L
        
        # Now R can be added to the window
        in_window[R] = True
        curr_len = R - L + 1
        if curr_len > best_len:
            best_len = curr_len
            best_start = L
            
        # Deadline guard
        if time.time() > DEADLINE:
            break

    # Verification step (optional but good for safety)
    # If best_len is 0, return a trivial valid interval if possible, though 
    # the problem guarantees a nonempty interval exists.
    if best_len == 0:
        # Find any valid x
        for x in range(1, N + 1):
            if not is_self_inv[x]:
                return [x, 1]
        return [1, 1] # Fallback, though 1 is self-inverse so this would be invalid if checked strictly, 
                      # but problem says "nonempty interval", and 1008 elements exist.
                      # Actually, if best_len is 0, it means no pair was found? 
                      # But single elements are valid unless self-inverse.
                      # The logic above handles single elements correctly.
                      # If R is not self-inverse, and inv[R] is not in window (empty), 
                      # we add R, len=1. So best_len should be at least 1.

    return [best_start, best_len]
