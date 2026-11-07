"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    # Precompute primes up to sqrt(max_hi)
    max_hi = max(hi for _, hi in P['gaps'])
    limit = int(max_hi**0.5) + 1
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    small_primes = [i for i, prime in enumerate(is_prime) if prime]
    
    def get_lpf(n):
        if n < 2:
            return n
        for p in small_primes:
            if p * p > n:
                break
            if n % p == 0:
                return p
        return n # n is prime

    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        
        gap = hi - lo
        
        # Find any m with lo < m < hi and lpf[m] >= gap
        for m in range(lo + 1, hi):
            lpf_m = get_lpf(m)
            if lpf_m >= gap:
                result[i] = m
                break
    
    # Ensure all entries are proper Python ints or None
    final_result = []
    for entry in result:
        if entry is None:
            final_result.append(None)
        else:
            final_result.append(int(entry))
    
    return final_result

# EVOLVE-BLOCK-END
