"""Search for n in [10000, 1000000] minimizing the largest prime factor of n*(n+1)."""
import time
import math

# EVOLVE-BLOCK-START

def _sieve_lpf(limit):
    """
    Return an array lpf where lpf[i] is the largest prime factor of i,
    for i in range(limit+1). Uses a segmented-style single pass.
    Memory: limit+1 integers. For limit=1000000 this is ~8 MB, fine.
    """
    lpf = list(range(limit + 1))
    lpf[0] = 0
    if limit >= 1:
        lpf[1] = 1
    # Sieve: for each prime p, update multiples
    i = 2
    while i * i <= limit:
        if lpf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if lpf[j] == j:  # not yet visited by a smaller prime
                    lpf[j] = i
        i += 1
    # Now for each number, compute its largest prime factor
    # lpf currently holds the smallest prime factor (SPF) for composite numbers,
    # and the number itself for primes.
    # We need the LARGEST prime factor.
    # Re-do: for each number, factorize using SPF to find LPF.
    # Actually, let's just compute LPF directly with a different approach:
    # For each prime p, for each multiple of p, set lpf[m] = max(lpf[m], p)
    # But that's O(n log log n) which is fine.
    lpf2 = [0] * (limit + 1)
    for i in range(2, limit + 1):
        if lpf2[i] == 0:  # i is prime
            for j in range(i, limit + 1, i):
                lpf2[j] = i  # since we iterate primes in increasing order, last one is largest
    lpf2[0] = 0
    lpf2[1] = 1
    return lpf2


def _largest_prime_factor_trial(n):
    """Exact trial division to find the largest prime factor of n."""
    if n <= 1:
        return 1
    lpf = 1
    d = 2
    while d * d <= n:
        while n % d == 0:
            lpf = d
            n //= d
        d += 1
    if n > 1:
        lpf = n
    return lpf


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N_MIN = 10000
    N_MAX = 1000000  # inclusive upper bound for n
    
    # We need largest prime factor of n*(n+1) for n in [10000, 1000000]
    # lpf(n*(n+1)) = max(lpf(n), lpf(n+1))
    # So we need lpf for numbers up to 1000001
    
    # Step 1: Compute largest prime factor for all numbers up to 1000001
    lpf_arr = _sieve_lpf(N_MAX + 1)
    
    # Step 2: Find n that minimizes max(lpf_arr[n], lpf_arr[n+1])
    best_n = N_MIN
    best_val = max(lpf_arr[N_MIN], lpf_arr[N_MIN + 1])
    
    for n in range(N_MIN + 1, N_MAX + 1):
        if time.time() > deadline - 5.0:
            break
        val = max(lpf_arr[n], lpf_arr[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
    
    # Step 3: Verify with exact trial division
    # Compute lpf(best_n) and lpf(best_n+1) via trial division
    lpf_n = _largest_prime_factor_trial(best_n)
    lpf_np1 = _largest_prime_factor_trial(best_n + 1)
    verified_val = max(lpf_n, lpf_np1)
    
    # If there's a mismatch, do a more careful search around the best region
    if verified_val != best_val:
        # Re-verify candidates with trial division in a window
        # Search a window around best_n for the true minimum
        window_start = max(N_MIN, best_n - 1000)
        window_end = min(N_MAX, best_n + 1000)
        for n in range(window_start, window_end + 1):
            if time.time() > deadline - 2.0:
                break
            ln = _largest_prime_factor_trial(n)
            ln1 = _largest_prime_factor_trial(n + 1)
            v = max(ln, ln1)
            if v < verified_val:
                verified_val = v
                best_n = n
    
    return best_n

# EVOLVE-BLOCK-END
