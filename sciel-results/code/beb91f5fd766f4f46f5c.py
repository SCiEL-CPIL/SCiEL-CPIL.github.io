# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    N_MIN = 10000
    N_MAX = 1000000
    limit = N_MAX + 1  # We need lpf up to N_MAX+1 = 1000001
    
    # Sieve of Eratosthenes to find all primes up to limit
    is_prime = np.ones(limit + 1, dtype=np.bool_)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i : limit+1 : i] = False
    
    primes = np.where(is_prime)[0]
    
    # Compute Largest Prime Factor (LPF) for each number up to limit
    # lpf[x] will store the largest prime factor of x
    lpf = np.zeros(limit + 1, dtype=np.int64)
    for p in primes:
        # For each multiple of p, set lpf[multiple] = p (since we iterate primes in increasing order,
        # the last prime that divides a number is its largest prime factor)
        for m in range(p, limit + 1, p):
            lpf[m] = p
    
    # For n in [N_MIN, N_MAX], we want to minimize max(lpf[n], lpf[n+1])
    # lpf[n] for n in [N_MIN, N_MAX] is lpf[N_MIN:N_MAX+1]
    # lpf[n+1] for n in [N_MIN, N_MAX] is lpf[N_MIN+1:N_MAX+2]
    
    lpf_n = lpf[N_MIN:N_MAX+1]       # lpf[n] for n = N_MIN, ..., N_MAX
    lpf_n1 = lpf[N_MIN+1:N_MAX+2]    # lpf[n+1] for n = N_MIN, ..., N_MAX
    
    # max of the two arrays element-wise
    max_lpf = np.maximum(lpf_n, lpf_n1)
    
    # Find all indices that achieve the minimum (in case of ties, pick smallest n)
    min_val = np.min(max_lpf)
    min_indices = np.where(max_lpf == min_val)[0]
    min_idx = min_indices[0]  # smallest n among ties
    
    best_n = N_MIN + min_idx
    best_val = int(min_val)
    
    # Verify with exact trial division (as per task requirement)
    def largest_prime_factor_exact(x):
        if x <= 1:
            return 1
        lp = 1
        f = 2
        while f * f <= x:
            while x % f == 0:
                lp = f
                x //= f
            f += 1
        if x > 1:
            lp = max(lp, x)
        return lp
    
    # Double-check our answer
    check_val = max(largest_prime_factor_exact(best_n), largest_prime_factor_exact(best_n + 1))
    if check_val != best_val:
        # Fallback: brute-force verify around best candidate
        for offset in range(-10, 11):
            candidate = best_n + offset
            if N_MIN <= candidate <= N_MAX:
                cv = max(largest_prime_factor_exact(candidate), largest_prime_factor_exact(candidate + 1))
                if cv < best_val:
                    best_n = candidate
                    best_val = cv
    
    return int(best_n)
# EVOLVE-BLOCK-END
