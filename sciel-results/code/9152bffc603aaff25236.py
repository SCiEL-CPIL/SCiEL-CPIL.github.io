"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    N = 1000000
    
    # Precompute smallest prime factor up to sqrt(N) for trial division
    sqrt_N = int(math.isqrt(N)) + 1
    spf = list(range(sqrt_N + 1))
    for i in range(2, int(math.isqrt(sqrt_N)) + 1):
        if spf[i] == i:
            for j in range(i * i, sqrt_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Generate list of primes up to sqrt_N
    primes = []
    for i in range(2, sqrt_N + 1):
        if spf[i] == i:
            primes.append(i)
    
    def largest_prime_factor(x):
        """Find the largest prime factor of x using trial division by precomputed primes."""
        if x <= 1:
            return 1
        result = 1
        for p in primes:
            if p * p > x:
                break
            while x % p == 0:
                result = p
                x //= p
        if x > 1:
            result = x
        return result
    
    min_val = float('inf')
    best_n = 10000
    
    # Search for n in [10000, 999999] such that max(lpf(n), lpf(n+1)) is minimized
    for n in range(10000, N):
        lp_n = largest_prime_factor(n)
        lp_n1 = largest_prime_factor(n + 1)
        val = lp_n if lp_n > lp_n1 else lp_n1
        
        if val < min_val:
            min_val = val
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
