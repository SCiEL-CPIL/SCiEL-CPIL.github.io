"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    
    # We want to minimize max(lpf(n), lpf(n+1))
    # Precompute primes up to sqrt(1000001) ~ 1000
    limit = 1001
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit+1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    
    def largest_prime_factor(n):
        """Compute largest prime factor of n using trial division by precomputed primes."""
        if n <= 1:
            return 1
        lpf = 1
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                lpf = p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            lpf = max(lpf, temp)
        return lpf
    
    best_n = 10000
    best_val = largest_prime_factor(10000)
    lp_n1 = largest_prime_factor(10001)
    best_val = max(best_val, lp_n1)
    
    # Scan through the range, checking each n
    # Use time budget to scan as many as possible
    for n in range(10001, 1000001):
        if time.time() > deadline:
            break
        
        lp_n = largest_prime_factor(n)
        if lp_n >= best_val:
            # Can't improve, skip n+1 check
            continue
        
        lp_np1 = largest_prime_factor(n + 1)
        if lp_np1 >= best_val:
            continue
        
        candidate_val = max(lp_n, lp_np1)
        if candidate_val < best_val:
            best_val = candidate_val
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
