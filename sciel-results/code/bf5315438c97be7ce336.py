import math
import time

def construct():
    start_time = time.time()
    limit = 10000
    # Safety deadline, well within the 90s limit
    deadline = start_time + 80.0

    # Generate primes up to 10000 using Sieve of Eratosthenes
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False

    primes = [i for i, flag in enumerate(is_prime) if flag]
    n = len(primes)

    if n == 0:
        return []
    if n == 1:
        return [primes[0]]

    # dp[i] = length of longest valid subsequence ending at primes[i]
    # prev[i] = index of previous element in that subsequence
    # last_gap[i] = the last gap in the subsequence ending at i (primes[i] - primes[prev[i]])
    
    dp = [1] * n
    prev = [-1] * n
    last_gap = [0] * n

    for i in range(1, n):
        pi = primes[i]
        best_len = 1
        best_prev = -1

        # Iterate j from i-1 down to 0.
        # This ensures that if we have a tie in dp[j], we keep the first one found
        # which corresponds to the largest j (smallest gap pi - primes[j]).
        # We only update if we find a strictly longer sequence.
        for j in range(i - 1, -1, -1):
            pj = primes[j]
            gap = pi - pj

            if dp[j] == 1:
                # A sequence of length 1 has no previous gap constraint (or gap 0).
                # We can always extend it to length 2.
                candidate_len = 2
            else:
                # Check non-decreasing gap constraint
                if gap < last_gap[j]:
                    continue
                candidate_len = dp[j] + 1

            if candidate_len > best_len:
                best_len = candidate_len
                best_prev = j
            # If candidate_len == best_len, we do NOT update best_prev.
            # Since we are iterating j downwards, the current best_prev has a larger index
            # than the current j, resulting in a smaller gap. This is the desired tie-break.

        dp[i] = best_len
        prev[i] = best_prev
        if best_prev != -1:
            last_gap[i] = pi - primes[best_prev]
        else:
            last_gap[i] = 0

        # Time check (should never trigger for N=1229)
        if time.time() > deadline:
            break

    # Find the index with the maximum length
    max_len = max(dp)
    # Among all indices with max length, prefer the one with smallest last_gap
    # This is technically not required for the length, but good practice.
    best_end = -1
    best_end_gap = float('inf')
    for i in range(n):
        if dp[i] == max_len:
            if last_gap[i] < best_end_gap:
                best_end_gap = last_gap[i]
                best_end = i

    # Backtrack to reconstruct the sequence
    seq = []
    curr = best_end
    while curr != -1:
        seq.append(primes[curr])
        curr = prev[curr]

    seq.reverse()
    return seq
