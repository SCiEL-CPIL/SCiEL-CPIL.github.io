import time

# EVOLVE-BLOCK-START

def construct():
    MOD = 1009
    N = MOD - 1  # 1008
    deadline = time.time() + 80  # safety margin before 90s kill

    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    best_start = 1
    best_len = 0
    
    # Two-pointer (sliding window) approach
    # e is the maximum valid end index for the current start s
    # As s increases, e can only increase or stay the same
    e = 0
    
    for s in range(1, MOD):
        if time.time() > deadline:
            break
            
        # Ensure e is at least s-1 so that the window [s, e] is potentially non-empty
        if e < s - 1:
            e = s - 1
            
        # Try to extend e as far as possible
        while e < N:
            x = e + 1
            # Check if adding x to [s, e] creates a conflict
            # Conflict exists if inv[x] is in [s, x]
            ix = inv[x]
            if ix < s or ix > x:
                e = x
            else:
                break
        
        # Update best
        if e >= s:
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s

    # Fallback (should not be needed if logic is correct, but safe)
    if best_len == 0:
        for s in range(1, MOD):
            if inv[s] != s:
                best_start = s
                best_len = 1
                break
    if best_len == 0:
        best_start = 1
        best_len = 1

    return [best_start, best_len]

# EVOLVE-BLOCK-END
