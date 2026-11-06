import time
import numpy as np

def construct():
    start = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill
    
    LIMIT = 1000000
    N = LIMIT + 1

    # 1. Compute the powerful-part array for every integer 0..N
    #    pp[k] = product of p^e over all primes p where p^e || k and e >= 2
    #    (i.e., p^e divides k but p^(e+1) does not)
    
    # Sieve primes up to sqrt(N)
    sqrt_N = int(np.sqrt(N)) + 1
    is_prime = np.ones(sqrt_N + 1, dtype=np.bool_)
    is_prime[:2] = False
    for i in range(2, int(np.sqrt(sqrt_N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.nonzero(is_prime)[0]

    pp = np.ones(N + 1, dtype=np.int64)

    # For each prime, process its powers
    for p in primes:
        p = int(p)
        pe = p * p
        while pe <= N:
            next_pe = pe * p
            
            if next_pe <= N:
                # Multiply pe into pp[k] for k where exact power of p is pe
                # k = m*pe, and m not divisible by p
                idx = np.arange(pe, N + 1, pe)
                mask_good = np.ones(len(idx), dtype=np.bool_)
                mask_good[p-1::p] = False
                pp[idx[mask_good]] *= pe
            else:
                # All multiples of pe have exact power pe (since pe*p > N)
                pp[pe::pe] *= pe
            
            pe = next_pe

    # 2. Score every candidate n in [10000, LIMIT]
    scores = pp[10000:LIMIT + 1] * pp[10001:N + 1]
    best_local = int(np.argmax(scores))
    best_n = 10000 + best_local

    return best_n
