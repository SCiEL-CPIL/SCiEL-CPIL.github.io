import time

def construct():
    start_time = time.time()
    MOD = 1009
    
    # Precompute inverses
    inv = [0] * (MOD + 1)
    for i in range(1, MOD):
        inv[i] = pow(i, MOD - 2, MOD)
        
    best_len = 0
    best_start = 1
    L = 1
    
    for R in range(1, MOD): # 1 to 1008
        if time.time() - start_time > 85:
            break
            
        I = inv[R]
        
        # If the inverse of R is currently in the window [L, R], 
        # the pair (I, R) is present. We must move L to exclude I.
        if L <= I <= R:
            L = I + 1
            
        if L <= R:
            current_len = R - L + 1
            if current_len > best_len:
                best_len = current_len
                best_start = L

    if best_len == 0:
        return [1, 1]

    return [best_start, best_len]
