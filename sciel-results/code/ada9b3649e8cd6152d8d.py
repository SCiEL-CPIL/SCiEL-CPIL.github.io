import time
import numpy as np

def construct():
    start = time.time()
    LIMIT = 1000000
    N = LIMIT + 1

    # 1. Compute the powerful-part array for every integer 0..N
    #    pp[k] = product of p^e over all prime powers p^e with e>=2 dividing k
    #    We use a segmented-style multiplicative sieve: for each prime p,
    #    for each power pe = p^e (e>=2), multiply pe into pp[k] for every
    #    k divisible by pe but NOT by p^(e+1).
    #    We only need primes up to N, but in practice only p^2 <= N matters
    #    for the "full power" contribution (p^2 divides k means e>=2 for p in k).
    #    Primes with p^2 > N have no k in [0,N] with e>=2 for p, so skip them.

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
            # All multiples of pe
            idx = np.arange(pe, N + 1, pe)  # indices divisible by pe
            # We want to multiply by pe ONLY for those NOT divisible by pe*p
            pe_next = pe * p
            if pe_next <= N:
                # Subtract those divisible by pe_next
                idx_bad = np.arange(pe_next, N + 1, pe_next)
                # Use set difference approach: mark
                # Simpler: use boolean mask
                mask = np.zeros(N + 1, dtype=np.bool_)
                mask[idx] = True
                mask[idx_bad] = False
                pp[mask] *= pe
            else:
                pp[idx] *= pe
            pe = pe_next

    # 2. Score every candidate n in [10000, LIMIT]
    #    score(n) = pp[n] * pp[n+1]
    #    n+1 ranges over [10001, LIMIT+1] = [10001, N]
    #    So we need pp[10000..LIMIT] and pp[10001..N]
    scores = pp[10000:LIMIT + 1] * pp[10001:N + 1]
    best_local = int(np.argmax(scores))
    best_n = 10000 + best_local

    # 3. Verify the answer and return
    return best_n
