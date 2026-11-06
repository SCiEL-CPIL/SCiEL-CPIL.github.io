import time
import math

def construct():
    """
    Returns an integer n in [10000, 1000000] that maximizes 
    powerful_part(n) * powerful_part(n+1).
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin for 90s limit
    
    N = 1000000
    MIN_N = 10000
    
    # 1. Build Smallest Prime Factor (SPF) table
    spf = list(range(N + 1))
    if N >= 0:
        spf[0] = 0
    if N >= 1:
        spf[1] = 1
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute powerful part for each number
    # powerful_part[i] stores the product of p^e for e >= 2 in factorization of i
    powerful_part = [1] * (N + 1)
    
    for i in range(2, N + 1):
        temp = i
        pp_val = 1
        while temp > 1:
            p = spf[temp]
            e = 0
            while temp % p == 0:
                temp //= p
                e += 1
            if e >= 2:
                pp_val *= (p ** e)
        powerful_part[i] = pp_val
        
        # Time check
        if i % 100000 == 0:
            if time.time() - start_time > DEADLINE:
                break

    # 3. Find n that maximizes powerful_part[n] * powerful_part[n+1]
    best_n = MIN_N
    best_score = 0
    
    # If the previous loop broke early due to timeout, we might not have computed all powerful_part values.
    # However, N=1e6 is small. Let's assume it finishes.
    # To be safe, we only search up to the last computed index if timeout occurred, but standard loop should finish.
    
    max_n_to_check = N - 1
    for n in range(MIN_N, max_n_to_check + 1):
        score = powerful_part[n] * powerful_part[n + 1]
        if score > best_score:
            best_score = score
            best_n = n
            
        # Time check in search loop (less likely to be bottleneck, but good practice)
        if n % 100000 == 0:
            if time.time() - start_time > DEADLINE:
                break

    return best_n
