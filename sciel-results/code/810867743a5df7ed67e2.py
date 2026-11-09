import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 1000000
    MAX_VAL = N + 1  # We need to access index N+1 for n+1 when n=N
    
    # 1. Sieve for Smallest Prime Factor (SPF)
    spf = list(range(MAX_VAL + 1))
    limit = int(MAX_VAL**0.5)
    for i in range(2, limit + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute powerful part for each number from 1 to MAX_VAL
    pp = [1] * (MAX_VAL + 1)
    
    # Local variables for speed
    spf_local = spf
    pp_local = pp
    
    for i in range(2, MAX_VAL + 1):
        x = i
        current_pp = 1
        while x > 1:
            p = spf_local[x]
            count = 0
            while x % p == 0:
                x //= p
                count += 1
            if count >= 2:
                current_pp *= (p ** count)
        pp_local[i] = current_pp
        
    # Check deadline
    if time.time() > deadline:
        return 10000
        
    # 3. Find n in [10000, 1000000] that maximizes pp[n] * pp[n+1]
    max_score = 0
    best_n = 10000
    
    for n in range(10000, N + 1):
        score = pp_local[n] * pp_local[n + 1]
        if score > max_score:
            max_score = score
            best_n = n
            
    return best_n
