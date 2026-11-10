def construct():
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for i in range(1, MOD):
        inv[i] = pow(i, MOD - 2, MOD)
        
    # min_R_at_L[L] = min { R_x | L_x == L }
    min_R_at_L = [float('inf')] * (MOD + 2)
    
    for x in range(1, MOD):
        y = inv[x]
        L = min(x, y)
        R = max(x, y)
        if R < min_R_at_L[L]:
            min_R_at_L[L] = R
            
    # Compute suffix minimums for min_R_from[s] = min { R_x | L_x >= s }
    min_R_from = [float('inf')] * (MOD + 2)
    min_R_from[MOD] = min_R_at_L[MOD] # L_x can be at most 1008? 
    # Wait, L_x is min(x, inv[x]). Max value is 1008.
    # So we only care up to 1008.
    
    # Iterate from 1008 down to 1
    current_min = float('inf')
    best_len = 0
    best_start = 1
    best_end = 1
    
    # We need to check all starts s from 1 to 1008
    # M(s) = min( min_R_from[s+1], min_R_at_L[s] )
    # Let's just compute M(s) on the fly or precompute an array.
    
    M = [0] * (MOD + 2)
    curr_min = float('inf')
    for s in range(MOD - 1, 0, -1):
        if min_R_at_L[s] < curr_min:
            curr_min = min_R_at_L[s]
        M[s] = curr_min
        
    # Now find max length
    for s in range(1, MOD):
        # M[s] is min R_x for L_x >= s
        max_e = M[s] - 1
        if max_e > MOD - 1:
            max_e = MOD - 1
            
        if max_e >= s:
            length = max_e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
                best_end = max_e
        else:
            # If max_e < s, no valid interval of length >= 1
            # But wait, if M[s] is inf, max_e is large.
            # If M[s] is small, it might be < s.
            pass
            
    return [best_start, best_len]
