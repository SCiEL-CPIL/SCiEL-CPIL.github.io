"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import math
    import numpy as np
    from collections import deque

    deadline = time.time() + 85.0

    # --- 1. Sieve of Eratosthenes to get all primes up to 10000 ---
    LIMIT = 10000
    sieve = bytearray([1]) * (LIMIT + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    primes = [i for i in range(2, LIMIT + 1) if sieve[i]]
    n_primes = len(primes)

    if n_primes < 3:
        return primes

    # --- 2. Build a directed graph: edge i->j (i<j) if primes[j]-primes[i] >= "prev gap" ---
    # The constraint is: for a sequence p_{a0}, p_{a1}, ..., p_{ak} with a0 < a1 < ... < ak,
    # we need (p_{a1}-p_{a0}) <= (p_{a2}-p_{a1}) <= ... <= (p_{ak}-p_{a{k-1}})
    # This is a longest path in a DAG with the nondecreasing gap constraint.

    # Strategy: Dynamic programming over the last two primes.
    # Let dp[i][j] = longest length of a valid sequence ending with primes[i], primes[j] (i<j)
    # where primes[j]-primes[i] is the last gap.
    # Transition: dp[i][j] = 1 + max(dp[h][i] for all h < i such that primes[i]-primes[h] <= primes[j]-primes[i])
    # Base: dp[i][j] >= 2 for all i < j (sequence of just two primes)
    # Answer: max over all dp[i][j] of dp[i][j], plus we can always have sequences of length 1 or 2.

    # But this is O(n^3) in the worst case. n_primes ~ 1229, so n^3 ~ 1.8e9 which is too slow.
    # We need to optimize.

    # Optimized DP: For each pair (i, j) with i < j, we want the max dp[h][i] over h < i 
    # with primes[i] - primes[h] <= primes[j] - primes[i].
    # Let gap_ji = primes[j] - primes[i]. We need primes[h] >= primes[i] - gap_ji.
    # Since primes are sorted, h must be in some range [lo, i-1] where primes[lo] >= primes[i] - gap_ji.
    # So we need the max of dp[h][i] for h in [lo, i-1].

    # We can process this by iterating j from 1 to n-1, and for each j, iterate i from 0 to j-1.
    # For each fixed i, as j increases, gap_ji = primes[j]-primes[i] increases, so lo decreases (more h's qualify).
    # This is tricky. Let's think differently.

    # Alternative: Iterate over the last gap value. But gaps can be large.

    # Let's try a different DP formulation:
    # Let dp[j] = longest valid sequence ending at prime index j.
    # But we need to know the previous gap. So dp[j] alone isn't enough.

    # Let's go back to dp[i][j] but optimize the inner max query.
    # For fixed i, as we vary j > i, we need max dp[h][i] for h in [lo(j,i), i-1] where lo(j,i) is the
    # smallest h such that primes[h] >= primes[i] - (primes[j] - primes[i]).
    # Since primes[j] increases with j, primes[i] - (primes[j]-primes[i]) decreases, so lo(j,i) is non-increasing in j.

    # We can precompute for each i, an array best[i][k] = max dp[h][i] for h in [0, k].
    # Then for a given (i, j), lo = lower_bound(primes, primes[i] - (primes[j]-primes[i])), 
    # and the answer is best[i][lo-1] if lo > 0, else 0... wait, h ranges from lo to i-1.
    # best[i][k] = max dp[h][i] for h in [0, k]. Then max for h in [lo, i-1] is not directly best[i][i-1] minus something.
    # We need a range max query structure.

    # Since n_primes ~ 1229, n^2 ~ 1.5e6. We can store dp as a 2D list of size n x n.
    # For the inner max, we can use a segment tree or sparse table for each fixed i.
    # But building a sparse table for each i is O(n^2 log n) ~ 1.5e6 * 11 ~ 1.6e7, which is fine.

    # Actually, let's just use a simple approach: for each fixed i, maintain a running max as we sweep j.
    # For fixed i, as j goes from i+1 to n-1, the threshold primes[i] - (primes[j]-primes[i]) decreases.
    # So the set of valid h's only grows. We can maintain a pointer lo_i that decreases, and keep track of
    # the max dp[h][i] for h in [current_lo, i-1].

    # Let me restructure:
    # dp[i][j] for i < j.
    # For each i from 0 to n-2:
    #   Initialize a data structure for h in [0, i-1] to query max dp[h][i] over h >= some threshold.
    #   For j from i+1 to n-1:
    #       gap = primes[j] - primes[i]
    #       min_prime_h = primes[i] - gap  # h must satisfy primes[h] >= min_prime_h
    #       lo = bisect_left(primes, min_prime_h, 0, i)
    #       # dp[i][j] = 2 + (max dp[h][i] for h in [lo, i-1] if lo < i, else 0)
    #       # Actually dp[i][j] = 2 if no prior, or 1 + max dp[h][i] for valid h.
    #       # Base: a sequence ending with (i,j) has length at least 2.
    #       # If there's a valid h, dp[i][j] = dp[h][i] + 1 where dp[h][i] is the length of sequence ending at (h,i).
    #       # Wait, dp[h][i] already counts the sequence ending with ...h, i. Adding j gives length dp[h][i]+1.
    #       # But the base case: a sequence of just two primes (i,j) has length 2. So dp[i][j] = max(2, 1 + max_dp_h_i).
    #       # Where max_dp_h_i = max dp[h][i] for h in [lo, i-1].

    # Let's use a Fenwick tree or segment tree for range max queries over the h-index for each fixed i.
    # But since we're processing j in increasing order and lo is non-increasing, we can just maintain a running max.

    # For fixed i:
    #   Let lo_ptr = i  (initially no h is valid)
    #   As j increases, lo decreases. When lo decreases, we add new h's to our consideration.
    #   We maintain max_val = max of dp[h][i] for all h that have been "added" (i.e., h >= current_lo and h < i).
    #   When lo decreases from old_lo to new_lo, we add h from new_lo to old_lo-1 to our max.

    # This gives O(n^2) total work for all i and j.

    # Let's implement this.
    import bisect

    # dp[i][j] = length of longest valid sequence ending with primes[i], primes[j], i < j.
    # We'll store dp as a list of lists. dp[i] is a list where dp[i][j] is defined for j > i.
    # Actually, let's just use a 2D array of size n x n, with dp[i][j] = 0 if i >= j.

    # Initialize dp to 0
    dp = [[0] * n_primes for _ in range(n_primes)]

    # Base case: any pair (i, j) with i < j has dp[i][j] >= 2
    # We'll fill in the transitions.

    best_length = 2 if n_primes >= 2 else (1 if n_primes >= 1 else 0)
    best_seq = None

    # We need to track the actual sequence, not just the length.
    # Let's store parent[i][j] = h that gives the max dp[h][i] for the transition to (i,j).
    parent = [[-1] * n_primes for _ in range(n_primes)]

    # Process in order of increasing j, then i? Or increasing i?
    # dp[i][j] depends on dp[h][i] for h < i. So if we process i in increasing order, 
    # then when we process (i,j), all dp[h][i] for h < i are already computed.
    # But dp[h][i] depends on dp[k][h] for k < h, which are computed earlier.
    # So processing i from 0 to n-2, and for each i, j from i+1 to n-1, works.

    for i in range(n_primes - 1):
        # For fixed i, we'll process j from i+1 to n_primes-1
        # Maintain lo_ptr and running max over dp[h][i] for h in [lo_ptr, i-1]
        lo_ptr = i  # initially no valid h
        running_max = 0
        running_max_h = -1

        for j in range(i + 1, n_primes):
            gap = primes[j] - primes[i]
            min_prime = primes[i] - gap
            # Find the smallest h < i such that primes[h] >= min_prime
            new_lo = bisect.bisect_left(primes, min_prime, 0, i)
            # Add h from new_lo to lo_ptr-1 to running_max
            while lo_ptr > new_lo:
                lo_ptr -= 1
                if dp[lo_ptr][i] > running_max:
                    running_max = dp[lo_ptr][i]
                    running_max_h = lo_ptr

            if running_max > 0:
                dp[i][j] = running_max + 1
                parent[i][j] = running_max_h
            else:
                dp[i][j] = 2
                parent[i][j] = -1

            if dp[i][j] > best_length:
                best_length = dp[i][j]

    # Reconstruct the best sequence
    if best_length <= 2:
        return primes[:best_length]

    # Find the (i, j) pair that gives best_length
    best_i, best_j = -1, -1
    for i in range(n_primes):
        for j in range(i + 1, n_primes):
            if dp[i][j] == best_length:
                best_i, best_j = i, j
                break
        if best_i != -1:
            break

    # Reconstruct backwards
    seq = [primes[best_j], primes[best_i]]
    cur_i, cur_j = best_i, best_j
    while parent[cur_i][cur_j] != -1:
        h = parent[cur_i][cur_j]
        seq.append(primes[h])
        cur_i, cur_j = h, cur_i

    seq.reverse()
    return seq

# EVOLVE-BLOCK-END
