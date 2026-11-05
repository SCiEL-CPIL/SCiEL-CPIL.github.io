import time

def construct():
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    best_start = 2
    best_len = 0
    start_time = time.time()
    
    # Iterate over all possible starting positions S
    # S must be in [2, 1007] to avoid self-inverse 1 and 1008
    for S in range(2, MOD):
        # Check time limit
        if time.time() - start_time > 55:
            break
        
        # If even extending to 1007 can't beat current best, skip
        if 1007 - S + 1 <= best_len:
            continue
            
        E = S - 1
        # Extend E as far as possible
        # E can go up to 1007 (since 1008 is self-inverse and invalid)
        while E < 1007:
            new_E = E + 1
            
            # Check if adding new_E creates a conflict
            # Conflict if inv(new_E) is in [S, E]
            inv_new_E = inv[new_E]
            
            if S <= inv_new_E <= E:
                break
            else:
                E = new_E
        
        # If we formed a valid interval [S, E]
        if E >= S:
            current_len = E - S + 1
            if current_len > best_len:
                best_len = current_len
                best_start = S
                
    return [best_start, best_len]
