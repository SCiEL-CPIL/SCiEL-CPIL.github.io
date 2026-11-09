def construct():
    import time
    start = time.time()
    deadline = start + 50
    
    N = 1000000
    WINDOW = 24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    MAX_VAL = N + WINDOW

    # Precompute sp
    lp = [0] * (MAX_VAL + 1)
    for p in PRIMES:
        for m in range(p, MAX_VAL + 1, p):
            if lp[m] == 0:
                lp[m] = p

    sp = [1] * (MAX_VAL + 1)
    for m in range(2, MAX_VAL + 1):
        p = lp[m]
        if p:
            sp[m] = sp[m // p] * p
    
    best_n = 1000
    best_distinct = WINDOW + 1
    
    freq = {}
    current_distinct = 0
    
    # Initialize window for n=1000: sp[1001] ... sp[1024]
    for i in range(1001, 1001 + WINDOW):
        val = sp[i]
        count = freq.get(val, 0)
        if count == 0:
            current_distinct += 1
        freq[val] = count + 1
        
    if current_distinct < best_distinct:
        best_distinct = current_distinct
        best_n = 1000
        
    sp_local = sp
    freq_local = freq
    for n in range(1001, N + 1):
        if time.time() > deadline:
            break
            
        # Remove sp[n]
        old_val = sp_local[n]
        count = freq_local[old_val]
        if count == 1:
            del freq_local[old_val]
            current_distinct -= 1
        else:
            freq_local[old_val] = count - 1
            
        # Add sp[n+WINDOW]
        new_val = sp_local[n + WINDOW]
        count = freq_local.get(new_val, 0)
        if count == 0:
            current_distinct += 1
        freq_local[new_val] = count + 1
        
        if current_distinct < best_distinct:
            best_distinct = current_distinct
            best_n = n
            if best_distinct == 1:
                break
                
    return best_n
