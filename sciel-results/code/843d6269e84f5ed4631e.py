import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    MOD = 1009
    N = MOD - 1  # 1008
    
    # Precompute modular inverses
    # inv[i] is the modular inverse of i modulo MOD
    inv = [0] * (MOD + 1)
    for i in range(1, MOD):
        inv[i] = pow(i, MOD - 2, MOD)
        
    # min_R_at_L[L] = min { R | (L, R) is an inverse pair with L <= R }
    # We only care about pairs where the smaller element is L.
    min_R_at_L = [float('inf')] * (MOD + 2)
    
    for x in range(1, MOD):
        y = inv[x]
        L = min(x, y)
        R = max(x, y)
        if R < min_R_at_L[L]:
            min_R_at_L[L] = R
            
    # M[s] = min { R | L >= s }
    # This is the smallest R such that there exists a pair (L, R) with L >= s.
    # If we start an interval at s, we cannot include any R that is part of a pair (L, R)
    # where L >= s. Because if L >= s and R <= e, then both L and R are in [s, e],
    # which is invalid.
    # So e must be < min { R | L >= s }.
    M = [0] * (MOD + 2)
    curr_min = float('inf')
    # Iterate s from MOD-1 down to 1
    for s in range(MOD - 1, 0, -1):
        if min_R_at_L[s] < curr_min:
            curr_min = min_R_at_L[s]
        M[s] = curr_min
        
    best_len = 0
    best_start = 1
    
    for s in range(1, MOD):
        # Max valid end is M[s] - 1
        max_e = M[s] - 1
        if max_e > MOD - 1:
            max_e = MOD - 1
            
        if max_e >= s:
            length = max_e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
    
    if best_len == 0:
        # Fallback, though theoretically there should always be a valid interval of length >= 1
        # e.g., [2, 2] if 2 is not self-inverse
        best_start = 2
        best_len = 1
        
    return [best_start, best_len]
