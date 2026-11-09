import time

def construct():
    start_time = time.time()
    LIMIT = 1009
    MAX_VAL = 1008
    
    # Precompute modular inverses
    inv = [0] * (LIMIT + 1)
    for i in range(1, LIMIT):
        inv[i] = pow(i, -1, LIMIT)
    
    # Two-pointer / sliding window approach
    # Window [s, e] must not contain both x and inv[x] for any x
    # 1 and 1008 are self-inverse, so exclude them
    
    best_len = 0
    best_start = 2
    
    in_window = [0] * (LIMIT + 1)
    
    s = 2
    e = 1
    
    while s <= MAX_VAL:
        # Try to extend e as far as possible
        while e + 1 <= MAX_VAL:
            candidate = e + 1
            # If inv[candidate] is already in the window, adding candidate creates a conflict
            if in_window[inv[candidate]] > 0:
                break
            e += 1
            in_window[e] += 1
        
        # Current valid window is [s, e]
        length = e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
        
        # Shrink from left
        in_window[s] -= 1
        s += 1
        
        # Safety time check
        if time.time() - start_time > 85:
            break
    
    return [best_start, best_len]
