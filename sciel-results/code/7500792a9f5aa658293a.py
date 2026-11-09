"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def _sieve(limit):
    """Return list of primes <= limit using Sieve of Eratosthenes."""
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    deadline = time.time() + 75.0  # safety margin before 90s limit
    
    # Generate all primes <= 10000
    primes = _sieve(10000)
    n = len(primes)
    
    # DP for longest non-decreasing gap subsequence
    # dp[i] = length of longest valid subsequence ending at primes[i]
    # prev[i] = index of previous element in that subsequence (for reconstruction)
    dp = [1] * n
    prev = [-1] * n
    
    for i in range(1, n):
        best_len = 1
        best_prev = -1
        for j in range(i):
            # gap from primes[j] to primes[i] must be >= gap from primes[prev[j]] to primes[j]
            gap_ji = primes[i] - primes[j]
            if j == 0:
                # If j is the first element, there's no prior gap constraint
                # Any gap is valid as the first gap
                if dp[j] + 1 > best_len:
                    best_len = dp[j] + 1
                    best_prev = j
            else:
                prev_gap = primes[j] - primes[prev[j]]
                if gap_ji >= prev_gap:
                    if dp[j] + 1 > best_len:
                        best_len = dp[j] + 1
                        best_prev = j
        dp[i] = best_len
        prev[i] = best_prev
    
    # Find the end index of the longest subsequence
    max_len = max(dp)
    end_idx = dp.index(max_len)
    
    # Reconstruct the subsequence
    result = []
    idx = end_idx
    while idx != -1:
        result.append(primes[idx])
        idx = prev[idx]
    result.reverse()
    
    return result
# EVOLVE-BLOCK-END
