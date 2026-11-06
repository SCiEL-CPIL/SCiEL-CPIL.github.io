import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 1000000
    LOW = 10000
    
    # First, get all primes up to sqrt(N) = 1000
    limit = int(math.isqrt(N)) + 1
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, limit + 1) if is_prime[p]]
    
    # Array to hold powerful part for each index in [LOW, N]
    size = N - LOW + 1
    pw = np.ones(size, dtype=np.int64)
    
    # For each prime p, for each power p^e (e>=2), find numbers in [LOW,N] 
    # divisible by p^e but NOT by p^(e+1), and multiply pw by p^e.
    for p in primes:
        pe = p * p  # p^2
        e = 2
        while pe <= N:
            pe_next = pe * p  # p^(e+1)
            
            # Multiples of pe in [LOW, N]
            first = ((LOW + pe - 1) // pe) * pe
            if first > N:
                break
            
            # All multiples of pe
            mults_pe = np.arange(first, N + 1, pe)
            idxs_pe = mults_pe - LOW
            
            # Filter: keep only those where m // pe is NOT divisible by p
            # i.e., exact exponent is e
            quotients = mults_pe // pe
            mask = (quotients % p) != 0
            idxs_pe = idxs_pe[mask]
            
            if len(idxs_pe) > 0:
                pw[idxs_pe] *= pe
            
            pe = pe_next
            e += 1
            if time.time() > deadline - 5.0:
                break
    
    # Now find the n in [LOW, N-1] that maximizes pw[n-LOW] * pw[n+1-LOW]
    products = pw[:-1] * pw[1:]
    best_idx = int(np.argmax(products))
    best_n = LOW + best_idx
    
    return best_n
