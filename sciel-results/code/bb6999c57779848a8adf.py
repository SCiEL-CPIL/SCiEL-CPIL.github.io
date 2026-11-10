"""Rewritten program for smallest largest prime factor of consecutive-integer products."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin under 90s wall-clock
    
    max_n = 1000000
    max_val = max_n + 1  # we need lpf up to 1000001
    
    # Sieve for largest prime factor up to max_val
    lpf = [0] * (max_val + 1)
    for i in range(2, max_val + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, max_val + 1, i):
                lpf[j] = i
    
    # Find n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    best_n = 10000
    best_val = float('inf')
    
    for n in range(10000, max_n + 1):
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
