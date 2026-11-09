"""Rewritten program: DP for globally optimal non-decreasing-gap prime subsequence."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill

    # Sieve of Eratosthenes for primes <= 10000
    limit = 10000
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = 0

    primes = [p for p in range(2, limit + 1) if is_prime[p]]
    n = len(primes)

    # DP: best_len[i] = max length of valid subsequence ending at primes[i]
    # best_gap[i] = the last gap in that optimal subsequence (for tie-breaking, prefer smaller)
    # prev[i] = index of predecessor in the optimal subsequence
    best_len = [1] * n
    best_gap = [0] * n   # last gap; 0 means "no predecessor"
    prev = [-1] * n

    # For each pair (j, i) with j < i, gap = primes[i] - primes[j]
    # Valid if gap >= best_gap[j] (non-decreasing gaps)
    # We want to maximize length. For ties in length, prefer smaller best_gap[i]
    # (smaller last gap is more permissive for future extensions).

    for i in range(1, n):
        pi = primes[i]
        for j in range(i):
            gap = pi - primes[j]
            if gap >= best_gap[j]:
                cand_len = best_len[j] + 1
                if cand_len > best_len[i] or (cand_len == best_len[i] and gap < best_gap[i]):
                    best_len[i] = cand_len
                    best_gap[i] = gap
                    prev[i] = j

        if time.time() > deadline:
            break

    # Find the index with maximum best_len
    best_idx = max(range(n), key=lambda i: best_len[i])

    # Reconstruct the sequence
    seq = []
    cur = best_idx
    while cur != -1:
        seq.append(primes[cur])
        cur = prev[cur]
    seq.reverse()

    return seq
# EVOLVE-BLOCK-END
