import numpy as np
import time

def construct():
    # Deadline guard: 90s limit, use 85s for safety
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Range
    LO = 10000
    HI = 1000000
    
    # We need LPF for n and n+1, so up to HI+1
    MAX_VAL = HI + 1
    
    # Sieve to find Largest Prime Factor for all numbers up to MAX_VAL
    # lpf[i] will store the largest prime factor of i
    lpf = np.zeros(MAX_VAL + 1, dtype=np.int32)
    
    # Initialize lpf for primes
    # We iterate p from 2 to MAX_VAL
    # If lpf[p] is 0, p is prime
    # Then we update multiples of p
    
    # This loop is efficient for N=1e6
    for p in range(2, MAX_VAL + 1):
        if lpf[p] == 0:
            # p is prime
            # Mark all multiples of p with p as a prime factor
            # Since we go in increasing order, the last assignment will be the largest prime factor
            for multiple in range(p, MAX_VAL + 1, p):
                lpf[multiple] = p
                
    # Check time
    if time.time() > deadline:
        # Fallback to a simple heuristic if sieve was too slow (unlikely for 1e6)
        return LO

    # We want to minimize max(lpf[n], lpf[n+1]) for n in [LO, HI]
    # Create arrays for the range
    # indices: LO, LO+1, ..., HI
    # lpf_n: lpf[LO] ... lpf[HI]
    # lpf_n1: lpf[LO+1] ... lpf[HI+1]
    
    lpf_n = lpf[LO:HI+1]
    lpf_n1 = lpf[LO+1:HI+2]
    
    # Compute the cost for each n
    costs = np.maximum(lpf_n, lpf_n1)
    
    # Find the index of the minimum cost
    min_idx = np.argmin(costs)
    
    # The corresponding n is LO + min_idx
    best_n = LO + min_idx
    
    # The problem asks to return n.
    # The evaluator will compute the score based on this n.
    return int(best_n)
