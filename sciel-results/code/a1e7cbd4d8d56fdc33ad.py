import time
import sys
from math import prod
import numpy as np

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    N = 1000000
    MAX_VAL = N + 24
    
    # Precompute the small part for all numbers up to MAX_VAL
    # small_part[x] will be the factor of x composed of primes in 'primes'
    # Using NumPy for vectorized operations
    
    sp = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    for p in primes:
        power = p
        while power <= MAX_VAL:
            # Vectorized: multiply sp[multiples of power] by p
            indices = np.arange(power - 1, MAX_VAL + 1, power)
            sp[indices] *= p
            power *= p
    
    # Now sp[i] contains the complete factor of i composed of primes strictly below 24.
    
    best_n = 1000
    min_distinct = 100  # Start high
    start_time = time.time()
    deadline = 85  # Leave 5s margin for overhead/return
    
    # Search for n minimizing the number of distinct values in {sp[n+1], ..., sp[n+24]}
    # Use NumPy for vectorized set operations where possible
    
    for n in range(1000, N + 1):
        if (n & 2047) == 0:
            if time.time() - start_time > deadline:
                break
        
        # Extract the 24 values and count distinct
        factors = sp[n+1 : n+25]
        distinct_count = len(np.unique(factors))
        
        if distinct_count < min_distinct:
            min_distinct = distinct_count
            best_n = n
            
            if min_distinct == 1:
                break
    
    return best_n

construct()
