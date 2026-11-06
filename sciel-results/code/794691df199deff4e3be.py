import time

# EVOLVE-BLOCK-START

def construct():
    MOD = 1009
    N = MOD - 1  # 1008
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
        
    best_start = 1
    best_len = 0
    
    # For each start s, find the maximum valid end e
    # such that for all x in [s, e], inv[x] is not in [s, e]
    for s in range(1, MOD):
        # Self-inverse elements cannot be in any valid interval
        if inv[s] == s:
            continue
        
        # limit tracks the maximum end index allowed by constraints
        limit = N
        max_valid_e = s - 1
        
        for x in range(s, MOD):
            ix = inv[x]
            
            # If inv[x] >= s, then for the interval to be valid,
            # we need e < ix (since ix >= s means ix could be in [s, e])
            if ix >= s:
                candidate_limit = ix - 1
                if candidate_limit < limit:
                    limit = candidate_limit
            
            # If x exceeds the allowed limit, stop
            if x > limit:
                break
            
            max_valid_e = x
        
        length = max_valid_e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
    
    # Fallback
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
