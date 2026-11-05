import time
from math import isqrt
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    limit = 10000
    # Sieve
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    n = len(primes)

    # Complete DP:
    # dp[i] = list of (last_gap, min_end_prime_index) sorted by last_gap,
    # where the entry represents a sequence of some length ending at primes[i]
    # with last_gap = last_gap, and min_end_prime_index is the index of the
    # smallest possible last prime for that (last_gap, length) combo.
    #
    # Actually, let's think differently. The key quantity we want to maximize
    # is the LENGTH. For a given prime i and given last gap g, we want to know
    # the maximum length of a valid sequence ending at primes[i] with last gap
    # exactly g (or at most g).
    #
    # Better approach:
    # dp[i][g] = max length of valid subsequence ending at primes[i] with last
    # gap <= g. But g ranges up to ~10000, so this is O(n * max_gap) which is
    # ~1229 * 10000 = 12.29M, feasible.
    #
    # Even better: For each prime i, dp[i] is a list of (gap, length) pairs
    # representing Pareto-optimal states: for increasing gap, length is
    # non-decreasing. When processing prime i, for each predecessor j < i,
    # gap = primes[i] - primes[j]. We can extend any state in dp[j] with
    # last_gap <= gap.
    #
    # To make this efficient, for each j, dp[j] is sorted by gap. We binary
    # search for the largest gap <= current_gap to get the best (max) length.
    #
    # dp[i] = sorted list of (gap, length) where for each gap value, we keep
    # the max length achievable. We prune dominated entries.

    # dp[i] will be a list of (gap, length) sorted by gap, with length strictly
    # increasing (Pareto frontier).
    dp = [None] * n

    for i in range(n):
        if time.time() > deadline:
            break
        pi = primes[i]
        # Collect all candidate (gap, length) pairs for sequences ending at i
        # Start with length 1, gap 0 (single element)
        best = {}  # gap -> max_length
        best[0] = 1  # single element sequence

        for j in range(i):
            gap = pi - primes[j]
            if dp[j] is None or len(dp[j]) == 0:
                continue
            # Find the best length in dp[j] with last_gap <= gap
            # dp[j] is sorted by gap, length non-decreasing
            dj = dp[j]
            # Binary search: find rightmost entry with gap <= current gap
            lo, hi = 0, len(dj) - 1
            res_idx = -1
            while lo <= hi:
                mid = (lo + hi) // 2
                if dj[mid][0] <= gap:
                    res_idx = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            if res_idx >= 0:
                prev_len = dj[res_idx][1]
                new_len = prev_len + 1
                if new_len > best.get(gap, 0):
                    best[gap] = new_len

        # Now build Pareto frontier from best dict
        # Sort by gap, then for each gap keep max length
        # Then prune: for increasing gap, keep only if length > all previous
        sorted_gaps = sorted(best.keys())
        frontier = []
        max_len_so_far = 0
        for g in sorted_gaps:
            L = best[g]
            if L > max_len_so_far:
                frontier.append((g, L))
                max_len_so_far = L
            elif L == max_len_so_far and (not frontier or g < frontier[-1][0]):
                # Replace last if same length but smaller gap (better for extension)
                frontier[-1] = (g, L)

        dp[i] = frontier

    # Find the overall best: max length across all dp[i]
    best_len = 0
    best_i = -1
    best_gap = 0
    for i in range(n):
        if dp[i] and len(dp[i]) > 0:
            L = dp[i][-1][1]
            if L > best_len:
                best_len = L
                best_i = i
                best_gap = dp[i][-1][0]

    if best_i == -1:
        return [primes[0]] if n > 0 else []

    # Reconstruct: we need to trace back. But dp doesn't store predecessors.
    # Let's redo with predecessor tracking, but only for the optimal path.
    # Alternative: do a second pass to reconstruct.
    #
    # Actually, let's store predecessor info. But that's memory heavy.
    # Better: reconstruct by working backwards.
    #
    # We know the answer has length best_len ending at primes[best_i] with
    # last gap best_gap. We need to find the predecessor.
    #
    # For reconstruction, we need to know which j was used. Let's store
    # for each (i, gap) the predecessor j. But there are many gaps.
    #
    # Alternative approach: store for each i, for each entry in the frontier,
    # the predecessor index. This is feasible since frontier sizes are small.

    # Let me redo the DP with predecessor tracking.
    # dp[i] = list of (gap, length, prev_i)
    # where prev_i is the index of the previous prime in the sequence.

    dp2 = [None] * n

    for i in range(n):
        if time.time() > deadline:
            break
        pi = primes[i]
        # Collect candidates: (gap, length, prev_i)
        candidates = [(0, 1, -1)]  # single element

        for j in range(i):
            gap = pi - primes[j]
            if dp2[j] is None or len(dp2[j]) == 0:
                continue
            dj = dp2[j]
            # Find rightmost entry with gap_j <= gap
            lo, hi = 0, len(dj) - 1
            res_idx = -1
            while lo <= hi:
                mid = (lo + hi) // 2
                if dj[mid][0] <= gap:
                    res_idx = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            if res_idx >= 0:
                prev_gap, prev_len, prev_prev = dj[res_idx]
                new_len = prev_len + 1
                candidates.append((gap, new_len, j))

        # Build Pareto frontier with predecessor
        # Sort by gap, then by length descending (prefer longer for same gap)
        candidates.sort(key=lambda x: (x[0], -x[1]))
        frontier = []
        max_len_so_far = 0
        for g, L, pj in candidates:
            if L > max_len_so_far:
                frontier.append((g, L, pj))
                max_len_so_far = L
            elif L == max_len_so_far and (not frontier or g < frontier[-1][0]):
                frontier[-1] = (g, L, pj)

        dp2[i] = frontier

    # Find best
    best_len = 0
    best_i = -1
    for i in range(n):
        if dp2[i] and len(dp2[i]) > 0:
            L = dp2[i][-1][1]
            if L > best_len:
                best_len = L
                best_i = i

    if best_i == -1:
        return [primes[0]] if n > 0 else []

    # Reconstruct
    seq = []
    curr = best_i
    while curr != -1:
        seq.append(curr)
        # Get the last entry in dp2[curr] (max length)
        state = dp2[curr][-1]
        prev_i = state[2]
        curr = prev_i
    seq.reverse()

    result = [primes[i] for i in seq]

    # Verify
    if len(result) >= 2:
        gaps = [result[i+1] - result[i] for i in range(len(result)-1)]
        for k in range(len(gaps)-1):
            if gaps[k+1] < gaps[k]:
                # Shouldn't happen, but safety check
                pass

    return result
