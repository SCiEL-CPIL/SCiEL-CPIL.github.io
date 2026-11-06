import time

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 80.0  # Safety margin
    
    MOD = 1009
    N = MOD - 1  # 1008
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
        
    best_start = 1
    best_len = 0
    
    # Iterate over all possible starts
    for s in range(1, MOD):
        if time.time() > deadline:
            break
            
        # If s is a self-inverse, it cannot be in any valid interval
        if inv[s] == s:
            continue
            
        limit = N  # Maximum allowed end index imposed by constraints
        max_valid_e = s - 1
        
        # Scan forward from s
        for x in range(s, MOD):
            ix = inv[x]
            
            # If the inverse of x is within the potential range [s, ...],
            # it imposes a constraint that e must be less than ix.
            if ix >= s:
                # e must be <= ix - 1
                if ix - 1 < limit:
                    limit = ix - 1
            
            # If the current x exceeds the allowed limit, we cannot include x
            if x > limit:
                break
            
            # x is valid to be included in the interval [s, x]
            max_valid_e = x
            
        length = max_valid_e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
            
    # Fallback if no interval found (should not happen as single elements
    # that are not self-inverses are valid)
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
