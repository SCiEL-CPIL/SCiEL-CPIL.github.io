"""Rewritten: longest prime subsequence <=10000 with nondecreasing gaps via DP on pairs."""
import time
from math import isqrt
from bisect import bisect_left

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    # Sieve primes up to 10000
    N = 10000
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

    # dp[i][j] for i<j: max length ending ...p_i,p_j. Use list of lists.
    # dp[i][j] stored in a 2D list; dp[i][j] = 2 baseline.
    dp = [[2] * P for _ in range(P)]  # dp[i][j] only valid for i<j
    best_len = 2
    best_pair = (0, 1)  # indices
    for i in range(1, P):
        # build suffix max over k in [0, i-1] of dp[k][i]
        suff = [0] * (i + 1)
        suff[i - 1] = dp[i - 1][i]
        for m in range(i - 2, -1, -1):
            v = dp[m][i]
            suff[m] = v if v >= suff[m + 1] else suff[m + 1]
        pi = primes[i]
        for j in range(i + 1, P):
            T = 2 * pi - primes[j]
            # find smallest k in [0, i-1] with primes[k] >= T
            k_min = bisect_left(primes, T, 0, i)
            if k_min < i:
                cand = 1 + suff[k_min]
            else:
                cand = 2
            dp[i][j] = cand
            if cand > best_len:
                best_len = cand
                best_pair = (i, j)
        if time.time() > deadline - 10:
            break
    # Reconstruct sequence from best_pair
    if best_len <= 2:
        return primes[:best_len] if best_len >= 2 else primes
    # backtrack
    seq = [primes[best_pair[1]], primes[best_pair[0]]]
    i, j = best_pair
    while len(seq) < best_len:
        target = dp[i][j] - 1
        gap_ij = primes[j] - primes[i]
        found = False
        for k in range(i - 1, -1, -1):
            if dp[k][i] == target and (primes[i] - primes[k]) <= gap_ij:
                seq.append(primes[k])
                j, i = i, k
                found = True
                break
        if not found:
            break
    seq.reverse()

    # Greedy extension: try to extend the sequence forward
    if len(seq) >= 2:
        last_gap = seq[-1] - seq[-2]
        last_val = seq[-1]
        # Find next prime > last_val with gap >= last_gap
        idx = bisect_left(primes, last_val + last_gap)
        if idx < P and time.time() < deadline - 5:
            seq.append(primes[idx])
            # Try to extend further
            while time.time() < deadline - 5:
                lg = seq[-1] - seq[-2]
                lv = seq[-1]
                nxt = bisect_left(primes, lv + lg)
                if nxt < P:
                    seq.append(primes[nxt])
                else:
                    break

    return seq
