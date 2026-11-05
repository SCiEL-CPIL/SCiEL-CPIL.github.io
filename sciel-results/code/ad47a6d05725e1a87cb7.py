import time

def construct():
    MOD = 1009
    N = 1008
    
    # Precompute inverses
    inv = [0] * (MOD)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
        
    # b_val[a] is the 'b' part of the pair {a, b} with a < b, or INF if no such pair exists for a
    # We only care about a in [2, 1007] because 1 and 1008 are self-inverse and forbidden
    INF = 10**9
    b_val = [INF] * (MOD + 1)
    
    for a in range(2, 1008):
        b = inv[a]
        if a < b:
            b_val[a] = b
        # if a > b, the pair is {b, a} with b < a, handled when index is b
        # if a == b, only 1 and 1008, which are excluded from this range
    
    # Compute suffix minimum of b_val
    # suffix_min[i] = min(b_val[i], b_val[i+1], ..., b_val[1007])
    suffix_min = [INF] * (MOD + 2)
    suffix_min[1008] = INF # 1008 is not in [2, 1007] range for 'a', but let's be safe
    for i in range(1007, 1, -1):
        suffix_min[i] = min(b_val[i], suffix_min[i+1])
        
    best_start = 2
    best_len = 0
    
    # Iterate over possible starting positions L
    # L cannot be 1 or 1008
    for L in range(2, 1008):
        # M is the minimum 'b' such that there is a pair {a, b} with a >= L and a < b
        M = suffix_min[L]
        
        # R must be < M to avoid containing the pair {a, M} where a >= L
        # Also R cannot be 1008 (self-inverse)
        # So R_max = min(1007, M - 1)
        
        R_max = min(1007, M - 1)
        
        if R_max >= L:
            length = R_max - L + 1
            if length > best_len:
                best_len = length
                best_start = L
                
    return [best_start, best_len]
