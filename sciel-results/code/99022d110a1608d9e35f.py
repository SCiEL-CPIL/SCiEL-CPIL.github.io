"""Precompute SPF via linear sieve, then scan for min largest prime factor of n(n+1)."""
import time
import math

# EVOLVE-BLOCK-START

def _build_spf(limit):
    """Linear sieve to compute smallest prime factor for all integers up to limit."""
    spf = list(range(limit + 1))
    if limit >= 0:
        spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    primes = []
    for i in range(2, limit + 1):
        if spf[i] == i:  # i is prime
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > limit:
                break
            spf[i * p] = p
    return spf

def _largest_prime_factor(n, spf):
    """Compute largest prime factor of n using SPF table."""
    if n <= 1:
        return 1
    lpf = 1
    while n > 1:
        p = spf[n]
        lpf = p
        while n % p == 0:
            n //= p
    return lpf

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    N_MAX = 1000000
    N_MIN = 10000

    # Build SPF up to N_MAX
    spf = _build_spf(N_MAX)

    best_n = N_MIN
    best_lpf = float('inf')

    # Scan all n in [N_MIN, N_MAX]
    for n in range(N_MIN, N_MAX + 1):
        lpf_n = _largest_prime_factor(n, spf)
        lpf_np1 = _largest_prime_factor(n + 1, spf)
        candidate_lpf = max(lpf_n, lpf_np1)
        if candidate_lpf < best_lpf:
            best_lpf = candidate_lpf
            best_n = n
        # Check time periodically (every 10000 iterations)
        if n % 10000 == 0:
            if time.time() > deadline:
                break

    return best_n

# EVOLVE-BLOCK-END
