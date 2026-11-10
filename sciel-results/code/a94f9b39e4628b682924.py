"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from bisect import bisect_left

def _sieve_primes(limit):
    """Return a numpy array of all primes <= limit."""
    if limit < 2:
        return np.array([], dtype=np.int64)
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.nonzero(sieve)[0].astype(np.int64)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill

    # Generate all primes <= 10000
    primes = _sieve_primes(10000)
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return [int(primes[0])]

    # dp[i] = length of longest valid sequence ending at prime index i
    # prev[i] = index of previous prime in the sequence ending at i
    dp = np.zeros(n, dtype=np.int32)
    prev = np.full(n, -1, dtype=np.int32)
    dp[0] = 1

    best_idx = 0
    best_len = 1

    for i in range(1, n):
        # We need to find j < i such that primes[i] - primes[j] >= primes[j] - primes[j-1]
        # i.e., the gap before j (if j>0) is <= gap from j to i
        # For j=0, there's no previous gap, so any i>0 can follow 0
        # Actually, let's reconsider: a sequence is a subsequence of primes p_{i0}, p_{i1}, ..., p_{ik}
        # with gaps g_k = p_{i_{k+1}} - p_{i_k} non-decreasing: g_0 <= g_1 <= ... <= g_{k-1}
        # 
        # dp[i] = max over all j < i where (if j > 0) primes[j]-primes[j-1] <= primes[i]-primes[j],
        #         or j == 0 (always valid as first element)
        #         of (dp[j] + 1)
        # But wait, this isn't quite right because the "previous gap" in the sequence ending at j
        # is not necessarily primes[j] - primes[j-1]; it's the gap between the two elements
        # in the optimal sequence ending at j.
        #
        # We need to track not just dp[i] but also the last gap used.
        # Let's use a different approach:
        # state[i][g] = max length of sequence ending at prime i with last gap exactly g?
        # That's too expensive.
        #
        # Alternative: For each i, consider all possible j < i. The constraint is:
        # if the sequence ending at j has last gap L_j, then we need L_j <= primes[i] - primes[j].
        # So we want max dp[j] over all j < i where last_gap[j] <= primes[i] - primes[j].
        #
        # We can process primes in order and maintain for each possible last_gap value the max dp.
        # But last_gap can be up to 10000, so we can use an array.
        #
        # Let's define: best_for_gap[g] = max dp[j] over all j processed so far where last_gap[j] == g
        # Then for prime i, we want max over g <= (primes[i] - primes[j]) ... but j varies.
        #
        # This is getting complex. Let me think differently.
        #
        # Actually, let's use the following: for each i, we iterate j from i-1 down to 0,
        # and check if the sequence ending at j can be extended to i.
        # The condition: if j == 0, always valid. If j > 0, we need last_gap[j] <= primes[i] - primes[j].
        #
        # To avoid O(n^2), we can note that for a fixed i, as j decreases, primes[i] - primes[j] increases.
        # So we want the max dp[j] among j < i where last_gap[j] <= primes[i] - primes[j].
        #
        # Let's just do O(n^2) but with early termination. n is about 1229 primes <= 10000.
        # 1229^2 ≈ 1.5M, which is fast in Python with some optimization.

        # For each i, find best j < i
        best_j = -1
        best_dp = 0
        p_i = primes[i]
        
        # j=0 is always a valid predecessor (sequence of length 1 -> length 2)
        best_dp = 1  # dp[0] = 1
        best_j = 0
        
        for j in range(1, i):
            if dp[j] + 1 <= best_dp:
                # Can't improve, but we still need to check because a later j might have larger dp
                # Actually no, we're looking for max dp[j]+1, so if dp[j]+1 <= best_dp, skip
                continue
            gap_prev = primes[j] - primes[j-1]
            gap_new = p_i - primes[j]
            if gap_new >= gap_prev:
                if dp[j] + 1 > best_dp:
                    best_dp = dp[j] + 1
                    best_j = j

        dp[i] = best_dp
        prev[i] = best_j
        
        if best_dp > best_len:
            best_len = best_dp
            best_idx = i

        # Time check
        if (i % 100 == 0) and (time.time() > deadline):
            break

    # Reconstruct the sequence
    seq = []
    idx = best_idx
    while idx != -1:
        seq.append(int(primes[idx]))
        idx = prev[idx]
    seq.reverse()

    return seq

# EVOLVE-BLOCK-END
