import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
        
    best_len = 0
    best_start = 1
    
    # Sliding window approach
    # left is the start of the current valid window
    # We iterate right from 1 to N
    left = 1
    in_window = [False] * (N + 1)
    
    for right in range(1, N + 1):
        # Check if 'right' creates a conflict
        # 1. Self-inverse
        if inv[right] == right:
            # 'right' cannot be in any valid interval
            # We must move left past 'right'
            # Also, we need to clear the window? 
            # Actually, if we just set left = right + 1, the window becomes empty.
            # But we need to ensure in_window is consistent.
            # It's easier to just reset the window state if it becomes invalid in a way that requires clearing many elements,
            # but here the logic is:
            # If inv[right] == right, then 'right' conflicts with itself.
            # So the window cannot contain 'right'.
            # So we move left to right + 1.
            # The elements in [left, right-1] were in the window.
            # We need to remove them from in_window?
            # Yes, because the window is now [right+1, right] which is empty.
            # So we should clear the window.
            # However, clearing the whole window is O(N).
            # We can just iterate and remove?
            # Since N is small, we can just do:
            for x in range(left, right):
                in_window[x] = False
            left = right + 1
            # The window is now empty.
            # No length update needed (length 0)
            continue
            
        # 2. Pair with existing element
        if in_window[inv[right]]:
            # Conflict with inv[right]
            # We must remove inv[right] and everything before it
            conflict_idx = inv[right]
            for x in range(left, conflict_idx + 1):
                in_window[x] = False
            left = conflict_idx + 1
            
        # Now, add 'right' to the window
        in_window[right] = True
        
        current_len = right - left + 1
        if current_len > best_len:
            best_len = current_len
            best_start = left
            
    # Fallback if no valid interval found (should not happen)
    if best_len == 0:
        for x in range(1, N + 1):
            if inv[x] != x:
                best_start = x
                best_len = 1
                break
                
    return [best_start, best_len]
