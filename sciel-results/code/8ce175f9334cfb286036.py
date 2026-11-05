"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin within 60s limit
    
    N = 1000000
    # Compute largest prime factor for each number up to N+1
    lpf = [0] * (N + 2)
    
    for i in range(2, N + 2):
        if lpf[i] == 0:  # i is prime
            for j in range(i, N + 2, i):
                lpf[j] = i
    
    # Check deadline
    if time.time() > deadline:
        return 10000
    
    # Find n in [10000, 999999] minimizing max(lpf[n], lpf[n+1])
    best_n = 10000
    best_val = max(lpf[10000], lpf[10001])
    
    for n in range(10000, 1000000):
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
            if time.time() > deadline:
                break
    
    return best_n
# EVOLVE-BLOCK-END
