def construct():
    MAX_N = 1000000
    MAX_M = MAX_N + 24
    primes_below_24 = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # 1. Generate all 23-smooth numbers up to MAX_M
    smooths = [1]
    i = 0
    while i < len(smooths):
        current = smooths[i]
        for p in primes_below_24:
            val = current * p
            if val > MAX_M:
                break
            smooths.append(val)
        i += 1
    
    smooths = sorted(set(smooths))
    val_to_id = {v: idx for idx, v in enumerate(smooths)}
    K = len(smooths)
    
    # 2. Compute smooth part for each number in [1, MAX_M]
    smooth_vals = [1] * (MAX_M + 1)
    for p in primes_below_24:
        for multiple in range(p, MAX_M + 1, p):
            val = 1
            temp = multiple
            while temp % p == 0:
                val *= p
                temp //= p
            if val > 1:
                smooth_vals[multiple] *= val

    # 3. Map to IDs
    ids = [0] * (MAX_M + 1)
    for i in range(1, MAX_M + 1):
        ids[i] = val_to_id[smooth_vals[i]]
        
    # 4. Sliding window
    best_n = 1000
    best_count = 25
    
    freq = [0] * K
    
    # Initialize window for n=1000: [1001, ..., 1024]
    for i in range(1001, 1000 + 25):
        freq[ids[i]] += 1
        
    distinct_count = sum(1 for c in freq if c > 0)
    best_count = distinct_count
    best_n = 1000
    
    for n in range(1001, MAX_N + 1):
        # Remove n, add n+24
        idx_remove = ids[n]
        idx_add = ids[n + 24]
        
        freq[idx_remove] -= 1
        if freq[idx_remove] == 0:
            distinct_count -= 1
            
        freq[idx_add] += 1
        if freq[idx_add] == 1:
            distinct_count += 1
            
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count == 1:
                break
                
    return best_n
