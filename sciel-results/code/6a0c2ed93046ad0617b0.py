"""Longest prime subsequence <=10000 with NONDECREASING gaps.

Combines the vectorized pair-DP of the current program (suffix-max over valid
predecessors, vectorized searchsorted for the gap constraint) with the explicit
parent-tracking of the pure-Python reference.  The parent array makes
reconstruction O(L) and exact, removing the fragile O(L*P) backtrack search.
"""
import time
from math import isqrt
import numpy as np


def construct():
    deadline = time.time() + 82.0
    N = 10000

    # ---- sieve ----
    is_prime = bytearray([1]) * (N + 1)
    is_prime[0] = is_prime[1] = 0
    for p in range(2, isqrt(N) + 1):
        if is_prime[p]:
            for m in range(p * p, N + 1, p):
                is_prime[m] = 0
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    P = len(primes)
    if P < 2:
        return primes
    primes_arr = np.array(primes, dtype=np.int32)

    # dp[i][j]  = length of longest valid subsequence whose last two terms are
    #             primes[i], primes[j] (i < j).  All initialised to 2.
    # parent[i][j] = index k < i that achieves dp[i][j] = dp[k][i] + 1  (-1 if none).
    dp = np.full((P, P), 2, dtype=np.int16)
    parent = np.full((P, P), -1, dtype=np.int16)

    best_len = 2
    best_i, best_j = 0, 1

    for i in range(1, P - 1):
        if time.time() > deadline - 12:
            break

        col = dp[:i, i]                      # dp[k][i] for k in [0, i-1]

        # suffix best value  and  suffix argmax (first index attaining it)
        suff_max = np.maximum.accumulate(col[::-1])[::-1]
        rev = col[::-1]
        rev_arg = np.arange(i - 1, -1, -1)
        rev_max = np.maximum.accumulate(rev)
        rev_which = np.where(rev >= rev_max, rev_arg, -1)
        suff_arg = np.maximum.accumulate(rev_which)[::-1]

        T_arr = 2 * primes[i] - primes_arr[i + 1:]
        k_min = np.searchsorted(primes_arr[:i], T_arr, side='left')

        valid = k_min < i
        safe = np.where(valid, k_min, 0)

        cand = np.full(T_arr.shape, 2, dtype=np.int16)
        par = np.full(T_arr.shape, -1, dtype=np.int16)
        if np.any(valid):
            cand[valid] = 1 + suff_max[safe[valid]]
            par[valid] = suff_arg[safe[valid]]

        dp[i, i + 1:] = cand
        parent[i, i + 1:] = par

        m = int(np.max(cand))
        if m > best_len:
            best_len = m
            best_i = i
            best_j = i + 1 + int(np.argmax(cand))

    # ---- exact O(L) reconstruction from the parent array ----
    if best_len <= 2:
        return primes[:best_len]

    seq = [primes[best_j], primes[best_i]]
    i_cur, j_cur = best_i, best_j
    for _ in range(best_len - 2):
        k = int(parent[j_cur, i_cur])
        if k < 0:
            break
        seq.append(primes[k])
        j_cur, i_cur = i_cur, k
    seq.reverse()

    # ---- safe greedy extension (nondecreasing gaps, still <= 10000) ----
    if len(seq) >= 2 and time.time() < deadline - 4:
        last_gap = seq[-1] - seq[-2]
        last_val = seq[-1]
        idx = int(np.searchsorted(primes_arr, last_val + last_gap, side='left'))
        if idx < P:
            seq.append(int(primes_arr[idx]))
            while time.time() < deadline - 2:
                lg = seq[-1] - seq[-2]
                lv = seq[-1]
                nxt = int(np.searchsorted(primes_arr, lv + lg, side='left'))
                if nxt < P:
                    seq.append(int(primes_arr[nxt]))
                else:
                    break

    return seq
