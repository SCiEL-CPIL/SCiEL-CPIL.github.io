"""Rewritten program: DP over primes to find longest subsequence with nondecreasing gaps."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def _sieve_primes(limit):
    """Return list of primes <= limit."""
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i, v in enumerate(is_prime) if v]

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    primes = _sieve_primes(10000)
    n = len(primes)
    if n == 0:
        return []
    if n == 1:
        return primes

    # DP: dp[i] = length of longest valid subsequence ending at primes[i]
    # prev[i] = index of previous element in that subsequence (or -1)
    dp = [1] * n
    prev = [-1] * n

    for i in range(1, n):
        best_len = 1
        best_prev = -1
        pi = primes[i]
        for j in range(i):
            gap = pi - primes[j]
            # Check nondecreasing: if subsequence ending at j has length >= 2,
            # the last gap must be <= gap
            if dp[j] >= 2:
                last_gap = primes[j] - primes[prev[j]]
                if last_gap <= gap:
                    if dp[j] + 1 > best_len:
                        best_len = dp[j] + 1
                        best_prev = j
            else:
                # dp[j] == 1, any gap is fine
                if dp[j] + 1 > best_len:
                    best_len = dp[j] + 1
                    best_prev = j

        dp[i] = best_len
        prev[i] = best_prev

        # Time check
        if time.time() > deadline:
            break

    # Find the end index with maximum dp value
    max_len = max(dp)
    end_idx = dp.index(max_len)

    # Reconstruct
    seq = []
    idx = end_idx
    while idx != -1:
        seq.append(primes[idx])
        idx = prev[idx]
    seq.reverse()

    return seq
# EVOLVE-BLOCK-END
