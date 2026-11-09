"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import bisect
from collections import defaultdict

def _sieve(limit):
    """Return numpy array of all primes <= limit using a standard sieve."""
    if limit < 2:
        return np.array([], dtype=np.int64)
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = False
    sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.where(sieve)[0]

def construct():
    deadline = time.time() + 85.0
    t_start = time.time()

    # All primes <= 10000
    primes = _sieve(10000)
    n = len(primes)
    if n == 0:
        return []
    if n == 1:
        return [int(primes[0])]

    p_list = primes.tolist()
    p_arr = primes

    # DP with parent tracking in a single pass:
    # dp[i] = list of (length, last_gap, parent_j) Pareto-optimal pairs for subsequences ending at i.
    # Pareto-optimal: no pair (l1,g1,p1) is dominated by (l2,g2,p2) where l2>=l1 and g2<=g1.
    # For reconstruction, we keep the parent that achieved each (length, min_gap) combination.

    dp = [None] * n
    dp[0] = [(1, 0, -1)]

    best_len = 1
    best_end = 0

    for i in range(1, n):
        if time.time() > deadline:
            break
        pi = p_list[i]
        candidates = [(1, 0, -1)]
        for j in range(i):
            gap_ij = pi - p_list[j]
            for (l, g, pj) in dp[j]:
                if gap_ij >= g:
                    candidates.append((l + 1, gap_ij, j))
        # Pareto-optimize: for each length, keep min gap (and its parent)
        len_to_best = {}
        for (l, g, pj) in candidates:
            if l not in len_to_best or g < len_to_best[l][0]:
                len_to_best[l] = (g, pj)
        # Remove dominated lengths: keep (l, g) only if no l' > l has g' <= g
        lengths_sorted = sorted(len_to_best.keys(), reverse=True)
        pareto = []
        best_gap = float('inf')
        for l in lengths_sorted:
            g, pj = len_to_best[l]
            if g < best_gap:
                pareto.append((l, g, pj))
                best_gap = g
        dp[i] = pareto
        if pareto and pareto[0][0] > best_len:
            best_len = pareto[0][0]
            best_end = i

    # Reconstruct
    if best_end == 0:
        return [p_list[0]]

    seq = [best_end]
    cur = best_end
    target_len = best_len
    while target_len > 1:
        found = False
        for (l, g, pj) in dp[cur]:
            if l == target_len:
                seq.append(pj)
                cur = pj
                target_len -= 1
                found = True
                break
        if not found:
            break
    seq.reverse()
    result = [p_list[i] for i in seq]

    # Verify
    if len(result) >= 2:
        gaps = [result[i+1] - result[i] for i in range(len(result)-1)]
        if not all(gaps[i] <= gaps[i+1] for i in range(len(gaps)-1)):
            result = [p_list[0]]

    return result

# EVOLVE-BLOCK-END
