"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    N = 1000000
    # Precompute primes up to sqrt(N) = 1000
    limit = 1000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    def get_lpf(x):
        """Returns the largest prime factor of x."""
        temp = x
        last_p = 1
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                last_p = p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            return temp
        return last_p

    min_val = float('inf')
    best_n = 10000
    
    # Search for n in [10000, 999999]
    # We can optimize by noting that if we find a value, we only need to update if we find a smaller one.
    # However, we must check all to be sure we found the global min.
    
    for n in range(10000, N):
        lpf_n = get_lpf(n)
        lpf_np1 = get_lpf(n + 1)
        val = lpf_n if lpf_n > lpf_np1 else lpf_np1
        
        if val < min_val:
            min_val = val
            best_n = n
            
    return best_n
# EVOLVE-BLOCK-END
