import time

def construct():
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    best_start = 1
    best_len = 0
    
    start_time = time.time()
    deadline = start_time + 50.0
    
    L = 1
    for R in range(1, N + 1):
        if time.time() > deadline:
            break
        
        inv_R = inv[R]
        if inv_R >= L and inv_R <= R:
            L = inv_R + 1
        
        current_len = R - L + 1
        if current_len > best_len:
            best_len = current_len
            best_start = L
    
    return [best_start, best_len]
