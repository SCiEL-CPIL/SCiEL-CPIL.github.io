import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # 85 seconds to be safe against 90s limit
    
    L = 10000
    R = 1000000
    
    # We need prime factors for numbers up to R + 1
    max_val = R + 1
    
    # Sieve of Eratosthenes to find Smallest Prime Factor (SPF)
    spf = list(range(max_val + 1))
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute largest prime factor for all numbers up to max_val
    # This avoids repeated factorization in the main loop
    lpf = [1] * (max_val + 1)
    for n in range(2, max_val + 1):
        # Walk the SPF chain to find the largest prime factor
        m = n
        max_p = 1
        while m > 1:
            p = spf[m]
            if p > max_p:
                max_p = p
            while m % p == 0:
                m //= p
        lpf[n] = max_p
    
    best_n = L
    best_val = float('inf')
    
    # Main search loop
    for n in range(L, R + 1):
        # Check time periodically
        if (n - L) % 10000 == 0 and time.time() > deadline:
            break
            
        current_max = max(lpf[n], lpf[n + 1])
        
        if current_max < best_val:
            best_val = current_max
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
