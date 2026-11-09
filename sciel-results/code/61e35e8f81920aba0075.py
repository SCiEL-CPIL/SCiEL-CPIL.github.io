"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # Step 1: Generate all primes <= 10000 using Sieve of Eratosthenes
    def sieve_primes(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes = sieve_primes(10000)
    n = len(primes)

    # DP with improved tie-breaking:
    # dp[i] = max length of valid subsequence ending at primes[i]
    # gap[i] = the last gap in that optimal subsequence (minimized among max-length options)
    # prev_idx[i] = index of previous element
    #
    # Key structural change: when multiple predecessors give the same max length,
    # we pick the one with the SMALLEST gap to primes[i]. This maximizes future
    # extendability because a smaller last gap means more primes can follow.

    dp = [1] * n
    prev_idx = [-1] * n
    gap = [0] * n  # gap[i] = primes[i] - primes[prev[i]] if prev[i] >= 0, else 0

    for i in range(n):
        best_len = 1
        best_prev = -1
        best_gap = 0  # gap from best_prev to primes[i]

        for j in range(i):
            candidate_gap = primes[i] - primes[j]
            # Can we extend the subsequence ending at j with primes[i]?
            # Need: candidate_gap >= gap[j]
            if candidate_gap >= gap[j]:
                new_len = dp[j] + 1
                if new_len > best_len:
                    best_len = new_len
                    best_prev = j
                    best_gap = candidate_gap
                elif new_len == best_len:
                    # Tie-breaking: prefer smaller gap to current prime
                    # This leaves more room for future extensions
                    if candidate_gap < best_gap:
                        best_prev = j
                        best_gap = candidate_gap

        dp[i] = best_len
        prev_idx[i] = best_prev
        gap[i] = best_gap

    # Check time
    if time.time() > DEADLINE:
        # Fallback: return greedy result
        values = [primes[0]]
        for p in primes[1:]:
            if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                values.append(p)
        return values

    # Step 3: Find the index with maximum dp value
    max_len = max(dp)
    # Among all indices with max_len, pick the one with smallest last gap
    # (for consistency with our tie-breaking philosophy)
    candidates = [i for i in range(n) if dp[i] == max_len]
    end_idx = min(candidates, key=lambda i: gap[i])

    # Step 4: Reconstruct the sequence
    result = []
    idx = end_idx
    while idx >= 0:
        result.append(primes[idx])
        idx = prev_idx[idx]
    result.reverse()

    return result
# EVOLVE-BLOCK-END
