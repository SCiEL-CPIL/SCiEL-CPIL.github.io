import numpy as np
from math import isqrt

def construct():
    N = 10000
    # Sieve of Eratosthenes
    is_prime = bytearray([1]) * (N + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for m in range(i*i, N+1, i):
                is_prime[m] = 0

    primes = [i for i in range(2, N+1) if is_prime[i]]
    P = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    # DP: dp[j][g] = max length of valid sequence ending at primes[j] with last gap exactly g
    # We want non-decreasing gaps, so previous gap h <= g
    # dp[j][g] = 1 + max_{i<j, primes[i]+g=primes[j]} ( max_{h<=g} dp[i][h] )
    #           = 1 + max_{i: primes[i] = primes[j]-g} ( prefix_max[i][g] )
    # where prefix_max[i][g] = max(dp[i][0], dp[i][1], ..., dp[i][g])

    # But we can simplify: for each j and each gap g such that primes[j]-g is a prime,
    # dp[j][g] = 1 + prefix_max[i][g] where i = index of (primes[j]-g)

    # We'll store dp as a list of arrays, each of size max_gap+1
    # But max_gap can be up to N=10000, and P ~ 1229, so 1229*10001 ~ 12M entries
    # Use int16 since max length is small

    max_gap = N
    # dp[j] is an array of length max_gap+1
    # prefix[j] = np.maximum.accumulate(dp[j])

    dp = np.zeros((P, max_gap + 1), dtype=np.int16)
    prefix = np.zeros((P, max_gap + 1), dtype=np.int16)

    # Initialize: single element sequence has gap 0 (or we can treat gap as 0 for start)
    # Actually, let's define dp[j][g] where g is the gap from the previous element.
    # For the first element, there's no previous, so we set dp[j][0] = 1 for all j
    # Then for j > i, if primes[j] - primes[i] = g, then dp[j][g] = max(dp[j][g], 1 + prefix[i][g])
    # where prefix[i][g] = max_{h<=g} dp[i][h]

    for j in range(P):
        dp[j, 0] = 1
        prefix[j, 0] = 1
        # For g > 0, prefix will be computed after dp row is filled

    for j in range(1, P):
        p_j = primes[j]
        # For each possible gap g, check if p_j - g is a prime
        # g ranges from 1 to p_j - 2 (since p_j - g must be >= 2)
        # But we can iterate over i < j and compute g = p_j - primes[i]
        for i in range(j):
            g = p_j - primes[i]
            if g <= max_gap:
                # prefix[i][g] is max_{h<=g} dp[i][h]
                val = 1 + prefix[i, g]
                if val > dp[j, g]:
                    dp[j, g] = val

        # Compute prefix for row j
        prefix[j] = np.maximum.accumulate(dp[j])

    # Find the best sequence length
    # The length of sequence ending at j with last gap g is dp[j][g]
    # Overall best is max over all j and g of dp[j][g]
    # But we need to reconstruct the sequence

    best_len = 0
    best_j = 0
    best_g = 0
    for j in range(P):
        row_max = prefix[j, max_gap]
        if row_max > best_len:
            best_len = row_max
            best_j = j
            # Find the g that achieves this max
            # prefix[j][g] = max_{h<=g} dp[j][h], so we want the smallest g where prefix[j][g] = row_max
            # Actually, we want any g where dp[j][g] = row_max
            g_idx = np.argmax(dp[j] == row_max)
            best_g = g_idx

    # Reconstruct
    seq = []
    j_curr = best_j
    g_curr = best_g
    while True:
        p_curr = primes[j_curr]
        seq.append(p_curr)
        if g_curr == 0:
            break
        p_prev = p_curr - g_curr
        i_curr = prime_to_idx[p_prev]
        # Find h <= g_curr such that dp[i_curr][h] = prefix[i_curr][g_curr] - ... 
        # Actually, dp[j_curr][g_curr] = 1 + prefix[i_curr][g_curr]
        # So prefix[i_curr][g_curr] = dp[j_curr][g_curr] - 1
        # We need h such that dp[i_curr][h] = prefix[i_curr][g_curr]
        # i.e., max_{h'<=h} dp[i_curr][h'] = prefix[i_curr][g_curr]
        # The smallest such h is what we want
        target = prefix[i_curr, g_curr]
        # Find smallest h where prefix[i_curr][h] = target
        # Since prefix is non-decreasing, we can binary search
        h = np.searchsorted(prefix[i_curr], target, side='left')
        g_curr = h
        j_curr = i_curr

    seq.reverse()
    return seq
