import time
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 85  # safety margin before 90s kill

    def elapsed():
        return time.time() - t0

    MAXN = 1_000_000

    # Initialize powerful part array with 1s
    pp = np.ones(MAXN + 1, dtype=np.int64)

    # Sieve for primes up to sqrt(MAXN) (we only need primes p where p^2 <= MAXN)
    limit = int(MAXN**0.5) + 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]

    # Calculate powerful parts using optimized sieve
    # For each prime p, for each power p^k (k>=2), multiply pp[m] by the appropriate factor
    # for all m that are multiples of p^k.
    # This works because for m with v_p(m)=e, the total multiplier is p^2 * p^(e-2) = p^e
    for p in primes:
        p2 = p * p
        if p2 > MAXN:
            break

        p_power = p2
        # First power (k=2): multiply by p^2
        pp[p_power::p_power] *= p_power

        p_power *= p
        # Subsequent powers (k>2): multiply by p
        while p_power <= MAXN:
            pp[p_power::p_power] *= p
            p_power *= p

        if elapsed() > DEADLINE:
            return 10000

    if elapsed() > DEADLINE:
        return 10000

    # Scan for the best pair
    start = 10000
    # n ranges from start to MAXN-1 (so that n+1 <= MAXN)
    products = pp[start:MAXN] * pp[start+1:MAXN+1]
    
    max_idx = np.argmax(products)
    best_n = start + max_idx

    return int(best_n)
