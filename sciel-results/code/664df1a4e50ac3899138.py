"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START

def construct():
    p = 1009
    # Precompute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Self-inverse values are 1 and p-1 (1008)
    forbidden = {1, p - 1}
    
    # Sliding window approach
    # in_window[i] is True if i is in the current valid interval [left, right]
    in_window = [False] * (p + 1)
    left = 1
    max_len = 0
    best_start = 1
    
    for right in range(1, p):
        if right in forbidden:
            # Clear the window up to right
            for i in range(left, right):
                in_window[i] = False
            left = right + 1
            continue
        
        target = inv[right]
        
        # Shrink window from the left until the conflict is resolved
        # The conflict is that 'target' is in the window
        # We remove elements from the left until 'target' is removed
        while in_window[target]:
            in_window[left] = False
            left += 1
            
        # Add right to the window
        in_window[right] = True
        
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
            best_start = left
            
    # If no valid interval found (should not happen as length 1 intervals 
    # excluding 1 and 1008 are always valid), fallback
    if max_len == 0:
        return [2, 1]
        
    return [best_start, max_len]

# EVOLVE-BLOCK-END
