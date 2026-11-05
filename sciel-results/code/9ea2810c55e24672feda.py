import time

def construct():
    MOD = 1009
    N = 1008
    T = 89 # Safety margin for 90s limit
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
        
    # in_window[i] is True if i is in the current window [L, R]
    in_window = [False] * (MOD + 1)
    
    L = 1
    R = 0
    
    best_len = 0
    best_start = 1
    
    t0 = time.time()
    
    # Iterate L from 1 to 1008
    while L <= N:
        if time.time() - t0 > T:
            break
            
        # Extend R as far as possible
        # We can add cand = R+1 if inv[cand] is not in [L, R]
        while R < N:
            cand = R + 1
            # Condition: inv[cand] must NOT be in the current window
            if not in_window[inv[cand]]:
                in_window[cand] = True
                R = cand
            else:
                break
                
        cur_len = R - L + 1
        if cur_len > best_len:
            best_len = cur_len
            best_start = L
            
        # Move L forward
        in_window[L] = False
        L += 1
        
    # If no valid interval was found (should not happen as single elements like 2 are valid),
    # return a default valid single element interval
    if best_len == 0:
        return [2, 1]
        
    return [best_start, best_len]
