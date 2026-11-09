"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    # Set a deadline well within the 90s limit to be safe
    deadline = start_time + 80.0
    
    N = 1000000
    limit = N + 1
    
    # lpf[i] will hold the largest prime factor of i
    lpf = [0] * (limit + 1)
    
    # Sieve for largest prime factor
    for i in range(2, limit + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, limit + 1, i):
                lpf[j] = i
                
    # If we ran out of time (unlikely), return a safe default
    if time.time() > deadline:
        return 10000

    best_n = 10000
    best_val = float('inf')
    
    # Scan for the minimum largest prime factor
    for n in range(10000, N + 1):
        val = max(lpf[n], lpf[n+1])
        if val < best_val:
            best_val = val
            best_n = n
            
    return best_n
# EVOLVE-BLOCK-END
