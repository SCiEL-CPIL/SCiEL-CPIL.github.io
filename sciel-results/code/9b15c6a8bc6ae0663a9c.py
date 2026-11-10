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
            
    # Compute suffix minimums: M[s] = min { min_R_at_L[L] | L >= s }
    M = [float('inf')] * (MOD + 2)
    curr_min = float('inf')
    for s in range(MOD - 1, 0, -1):
        if min_R_at_L[s] < curr_min:
            curr_min = min_R_at_L[s]
        M[s] = curr_min
        
    # Now find max length
    best_len = 0
    best_start = 1
    best_end = 1
    
    for s in range(1, MOD):
        # M[s] is min R_x for L_x >= s
        # The interval [s, e] is valid iff e < M[s]
        # So max valid e is M[s] - 1
        max_e = M[s] - 1
        if max_e > MOD - 1:
            max_e = MOD - 1
            
        if max_e >= s:
            length = max_e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
                best_end = max_e
    
    return [best_start, best_len]
