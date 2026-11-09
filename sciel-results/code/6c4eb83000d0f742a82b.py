import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N_MIN = 10000
    N_MAX = 1000000
    MAX_N = N_MAX + 1  # We need n and n+1, so up to 1000001
    
    # Sieve for smallest prime factor - optimized
    spf = list(range(MAX_N + 1))
    for i in range(2, int(MAX_N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number
    powerful_part = [1] * (MAX_N + 1)
    
    for i in range(2, MAX_N + 1):
        n = i
        pp = 1
        while n > 1:
            p = spf[n]
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            if e >= 2:
                pp *= (p ** e)
        powerful_part[i] = pp
        
    # Check if we're still within time budget
    if time.time() > deadline - 5:
        # Fallback: return a reasonable default
        return 10000
    
    # Find the best n in [N_MIN, N_MAX] that maximizes powerful_part[n] * powerful_part[n+1]
    best_n = N_MIN
    best_val = powerful_part[N_MIN] * powerful_part[N_MIN + 1]
    
    for n in range(N_MIN + 1, N_MAX + 1):
        val = powerful_part[n] * powerful_part[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        # Periodic time check every 100k iterations
        if n % 100000 == 0 and time.time() > deadline - 2:
            break
    
    return best_n
