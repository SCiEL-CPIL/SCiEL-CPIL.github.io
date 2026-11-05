"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def construct():
    deadline = time.time() + 50  # safety margin before 60s limit
    N = 10000

    # Generate all primes up to N using sieve
    sieve = bytearray(b'\x01') * (N + 1)
    sieve[0] = 0
    sieve[1] = 0
    for i in range(2, isqrt(N) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes = [i for i in range(2, N + 1) if sieve[i]]
    n = len(primes)

    # dp[i] = length of longest valid subsequence ending at primes[i]
    # gap[i] = the last gap in that subsequence (i.e., primes[i] - prev_prime)
    # For the Fenwick tree approach:
    #   When processing prime i with value p_i, for each possible predecessor j < i,
    #   we need p_i - p_j >= gap[j].
    #   We want max dp[j] over all j where gap[j] <= p_i - p_j.
    #
    # Alternative cleaner DP:
    #   dp[i] = 1 + max{ dp[j] : j < i and (p_i - p_j) >= gap[j] }
    #   gap[i] = p_i - p_best_j (where p_best_j is the prime that gave the max)
    #
    # This is O(n^2) which for n=1229 is ~1.5M operations - fast enough.
    # Let's just do the O(n^2) DP directly.

    dp = [1] * n
    prev = [-1] * n  # prev[i] = index of predecessor in best chain ending at i

    for i in range(1, n):
        pi = primes[i]
        best_len = 1
        best_j = -1
        for j in range(i):
            gap_ij = pi - primes[j]
            if gap_ij >= (primes[j] - primes[prev[j]]) if prev[j] >= 0 else True:
                if dp[j] > best_len:
                    best_len = dp[j] + 1
                    best_j = j
        dp[i] = best_len
        prev[i] = best_j

    # Find the index with maximum dp
    max_len = 0
    end_idx = 0
    for i in range(n):
        if dp[i] > max_len:
            max_len = dp[i]
            end_idx = i

    # Reconstruct the sequence
    result = []
    idx = end_idx
    while idx >= 0:
        result.append(primes[idx])
        idx = prev[idx]
    result.reverse()

    # Verify and return
    assert all(result[i] <= N for i in range(len(result)))
    assert all(result[i] < result[i+1] for i in range(len(result)-1))
    for i in range(len(result)-1):
        assert sieve[result[i]] and sieve[result[i+1]]

    return result
# EVOLVE-BLOCK-END
