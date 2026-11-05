"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 1000000
    # Linear sieve to compute smallest prime factor (SPF) up to N+1
    spf = list(range(N + 2))
    spf[0] = 0
    spf[1] = 1
    primes = []
    for i in range(2, N + 2):
        if spf[i] == i:
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > N + 1:
                break
            spf[i * p] = p
    
    def lpf(x):
        """Return largest prime factor of x using SPF table."""
        if x <= 1:
            return 1
        m = 1
        while x > 1:
            p = spf[x]
            m = p
            while x % p == 0:
                x //= p
        return m
    
    best_n = 10000
    best_val = lpf(10000) * lpf(10001)  # actually we need max, not product
    # Wait, the task says "minimize the largest prime factor of n(n+1)".
    # The largest prime factor of a product is the max of the largest prime factors of the factors.
    # So we want to minimize max(lpf(n), lpf(n+1)).
    
    best_val = max(lpf(10000), lpf(10001))
    best_n = 10000
    
    for n in range(10001, N + 1):
        if time.time() > deadline:
            break
        
        ln = lpf(n)
        ln1 = lpf(n + 1)
        curr_max = ln if ln > ln1 else ln1
        
        if curr_max < best_val:
            best_val = curr_max
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
