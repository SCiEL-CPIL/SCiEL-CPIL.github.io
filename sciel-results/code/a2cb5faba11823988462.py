"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import math
    import time

    start_time = time.time()
    limit = 10000
    deadline = start_time + 85  # safety margin before 90s kill

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

    # DP approach with optimal tie-breaking:
    # dp[i] = length of longest valid subsequence ending at primes[i]
    # prev[i] = index of previous element in that subsequence
    # last_gap[i] = the last gap in the subsequence ending at i
    #
    # Key insight: when multiple predecessors give the same dp length,
    # we want the one with the SMALLEST last gap to maximize future extensions.
    # Smallest last gap = primes[i] - primes[j] is minimized when primes[j] is maximized,
    # i.e., j is as large as possible.

    dp = [1] * n
    prev = [-1] * n
    last_gap = [0] * n  # last_gap[i] = primes[i] - primes[prev[i]] if prev[i] != -1, else 0

    for i in range(1, n):
        pi = primes[i]
        best_len = 1
        best_prev = -1
        best_gap = 0  # we want to minimize the last gap among candidates with max length

        for j in range(i - 1, -1, -1):
            pj = primes[j]
            gap = pi - pj

            if dp[j] == 1:
                # Can always extend a length-1 sequence
                candidate_len = 2
                candidate_gap = gap
            else:
                # Check non-decreasing gap constraint
                if gap < last_gap[j]:
                    continue
                candidate_len = dp[j] + 1
                candidate_gap = gap

            if candidate_len > best_len:
                best_len = candidate_len
                best_prev = j
                best_gap = candidate_gap
            elif candidate_len == best_len and best_prev != -1:
                # Tie: prefer the candidate with the smaller gap for better future extensions
                if candidate_gap < best_gap:
                    best_prev = j
                    best_gap = candidate_gap
            elif candidate_len == best_len and best_prev == -1:
                best_prev = j
                best_gap = candidate_gap

        dp[i] = best_len
        prev[i] = best_prev
        last_gap[i] = pi - primes[best_prev] if best_prev != -1 else 0

        # Time check
        if time.time() > deadline:
            break

    # Find the index with the maximum length
    max_len = max(dp)
    # Among all indices with max length, prefer the one with smallest last_gap
    # for potential further extension (though we're done, this is for consistency)
    best_end = -1
    best_end_gap = float('inf')
    for i in range(n):
        if dp[i] == max_len:
            if last_gap[i] < best_end_gap:
                best_end_gap = last_gap[i]
                best_end = i
            elif last_gap[i] == best_end_gap:
                # Tie: prefer the larger prime (later index) for consistency
                if best_end == -1 or i > best_end:
                    best_end = i

    # Backtrack to reconstruct the sequence
    seq = []
    curr = best_end
    while curr != -1:
        seq.append(primes[curr])
        curr = prev[curr]

    seq.reverse()
    return seq
# EVOLVE-BLOCK-END
