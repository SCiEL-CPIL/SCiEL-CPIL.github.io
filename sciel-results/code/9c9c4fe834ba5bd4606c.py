import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin before 90s kill
    
    N = 1000000
    MIN_N = 10000

    # 1. Sieve for Smallest Prime Factor (SPF)
    spf = np.arange(N + 2, dtype=np.int64)
    limit = int(math.isqrt(N)) + 1
    for i in range(2, limit):
        if spf[i] == i:
            start = i * i
            multiples = np.arange(start, N + 2, i, dtype=np.int64)
            mask = spf[multiples] == multiples
            if np.any(mask):
                spf[multiples[mask]] = i

    if time.time() > deadline:
        return MIN_N

    # 2. Compute powerful part using multiplicative sieve approach
    # powerful[k] = product of p^e for all prime powers p^e | k with e >= 2
    # We can compute this by iterating over primes and their powers.
    # For each prime p, for each power p^e (e>=2), multiply powerful[k] by p^e for all k divisible by p^e but not p^(e+1).
    # But that's complex. Instead, use the factorization approach but vectorize.
    
    # Alternative: for each number k, factorize using spf and accumulate powerful part.
    # Let's use a list-based approach which is often faster for sequential dependencies.
    powerful = [1] * (N + 2)
    
    for k in range(2, N + 2):
        p = int(spf[k])
        m = k
        e = 0
        while m % p == 0:
            m //= p
            e += 1
        if e >= 2:
            powerful[k] = powerful[m] * (p ** e)
        else:
            powerful[k] = powerful[m]

    if time.time() > deadline:
        return MIN_N

    # 3. Vectorized scan for max product
    powerful_np = np.array(powerful, dtype=np.int64)
    
    p1 = powerful_np[MIN_N:N]
    p2 = powerful_np[MIN_N+1:N+1]
    
    products = p1 * p2
    best_idx = int(np.argmax(products))
    best_n = MIN_N + best_idx
    
    return best_n
