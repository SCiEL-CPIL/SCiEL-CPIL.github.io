import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    
    MAX_VAL = N_MAX + WINDOW + 10
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    S = [1] * (MAX_VAL + 1)
    
    # Precompute S
    for p in primes:
        p_pow = p
        # We can optimize by iterating powers
        # But simple loop is okay
        
        for m in range(p, MAX_VAL + 1, p):
            # We need to multiply S[m] by p^k
            # where k is the exponent of p in m
            # We can do this by checking divisibility
            # To avoid while loop overhead, we can precompute powers?
            # No, m varies.
            
            # Optimization: 
            # If m is not divisible by p*p, k=1.
            # If m is divisible by p*p, k>=2.
            
            # Let's stick to the while loop, it's robust.
            temp = m
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            if count:
                S[m] *= (p ** count)
                
    best_n = N_MIN
    best_count = float('inf')
    
    # Initial window
    counts = {}
    for i in range(N_MIN + 1, N_MIN + 1 + WINDOW):
        v = S[i]
        counts[v] = counts.get(v, 0) + 1
        
    current_distinct = len(counts)
    best_count = current_distinct
    best_n = N_MIN
    
    if best_count == 1:
        return best_n
        
    for n in range(N_MIN + 1, N_MAX + 1):
        # Remove S[n]
        v_out = S[n]
        c = counts[v_out] - 1
        if c == 0:
            del counts[v_out]
        else:
            counts[v_out] = c
            
        # Add S[n + WINDOW]
        v_in = S[n + WINDOW]
        counts[v_in] = counts.get(v_in, 0) + 1
        
        current_distinct = len(counts)
        
        if current_distinct < best_count:
            best_count = current_distinct
            best_n = n
            if best_count == 1:
                return best_n
                
        if time.time() > deadline:
            break
            
    return best_n
