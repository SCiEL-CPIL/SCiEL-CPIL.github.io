"""Rewritten: longest prime subsequence <=10000 with nondecreasing gaps via DP on pairs."""
import time
from math import isqrt
from bisect import bisect_left, bisect_right
import numpy as np

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

    # dp[i][j] for i<j: max length ending ...p_i,p_j. Use numpy 2D array.
    dp = np.full((P, P), 2, dtype=np.int16)
    best_len = 2
    best_pair = (0, 1)
    primes_arr = np.array(primes, dtype=np.int32)
    
    for i in range(1, P):
        # build suffix max over k in [0, i-1] of dp[k][i]
        col_i = dp[:i, i]
        suff = np.maximum.accumulate(col_i[::-1])[::-1]
        pi = primes[i]
        pi_arr = primes_arr[i]
        for j in range(i + 1, P):
            T = 2 * pi - primes[j]
            # find smallest k in [0, i-1] with primes[k] >= T
            k_min = bisect_left(primes, T, 0, i)
            if k_min < i:
                cand = 1 + int(suff[k_min])
            else:
                cand = 2
            dp[i, j] = cand
            if cand > best_len:
                best_len = cand
                best_pair = (i, j)
        if time.time() > deadline - 15:
            break
    
    # Reconstruct sequence from best_pair
    if best_len <= 2:
        return primes[:best_len] if best_len >= 2 else primes
    # backtrack
    seq = [primes[best_pair[1]], primes[best_pair[0]]]
    i, j = best_pair
    while len(seq) < best_len:
        target = int(dp[i, j]) - 1
        gap_ij = primes[j] - primes[i]
        found = False
        for k in range(i - 1, -1, -1):
            if int(dp[k, i]) == target and (primes[i] - primes[k]) <= gap_ij:
                seq.append(primes[k])
                j, i = i, k
                found = True
                break
        if not found:
            break
    seq.reverse()

    # Greedy extension: try to extend the sequence forward
    if len(seq) >= 2 and time.time() < deadline - 10:
        last_gap = seq[-1] - seq[-2]
        last_val = seq[-1]
        idx = bisect_left(primes, last_val + last_gap)
        if idx < P:
            seq.append(primes[idx])
            while time.time() < deadline - 5:
                lg = seq[-1] - seq[-2]
                lv = seq[-1]
                nxt = bisect_left(primes, lv + lg)
                if nxt < P:
                    seq.append(primes[nxt])
                else:
                    break

    # Enhanced local search: try to extend and improve
    if len(seq) >= 2 and time.time() < deadline - 5:
        def try_extend(s, max_time=None):
            """Try to greedily extend sequence s, return extended version."""
            if max_time is None:
                max_time = deadline - 3
            s = list(s)
            if len(s) >= 2:
                last_gap = s[-1] - s[-2]
                last_val = s[-1]
                idx = bisect_left(primes, last_val + last_gap)
                if idx < P:
                    s.append(primes[idx])
                    while time.time() < max_time:
                        lg = s[-1] - s[-2]
                        lv = s[-1]
                        nxt = bisect_left(primes, lv + lg)
                        if nxt < P:
                            s.append(primes[nxt])
                        else:
                            break
            return s

        def local_improve(s, max_time=None):
            """Try to improve interior positions of sequence s."""
            if max_time is None:
                max_time = deadline - 3
            s = list(s)
            improved = True
            while improved and time.time() < max_time:
                improved = False
                for pos in range(1, len(s) - 1):
                    prev_gap = s[pos] - s[pos-1]
                    next_gap = s[pos+1] - s[pos]
                    lo = s[pos-1] + prev_gap
                    hi = s[pos+1] - next_gap
                    if lo > hi:
                        continue
                    idx_lo = bisect_left(primes, lo)
                    idx_hi = bisect_right(primes, hi)
                    for new_idx in range(idx_lo, idx_hi):
                        new_val = primes[new_idx]
                        if new_val == s[pos]:
                            continue
                        new_prev_gap = new_val - s[pos-1]
                        new_next_gap = s[pos+1] - new_val
                        if new_prev_gap >= prev_gap and new_next_gap >= next_gap:
                            s[pos] = new_val
                            improved = True
                            break
                    if improved:
                        break
            return s

        # First, try to extend the current sequence
        seq = try_extend(seq)
        
        # Now do local improvement + extension iterations
        best_seq = list(seq)
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            # Try improving each interior position
            for pos in range(1, len(seq) - 1):
                if time.time() > deadline - 4:
                    break
                prev_gap = seq[pos] - seq[pos-1]
                next_gap = seq[pos+1] - seq[pos]
                lo = seq[pos-1] + prev_gap
                hi = seq[pos+1] - next_gap
                if lo > hi:
                    continue
                idx_lo = bisect_left(primes, lo)
                idx_hi = bisect_right(primes, hi)
                for new_idx in range(idx_lo, idx_hi):
                    new_val = primes[new_idx]
                    if new_val == seq[pos]:
                        continue
                    new_prev_gap = new_val - seq[pos-1]
                    new_next_gap = seq[pos+1] - new_val
                    if new_prev_gap >= prev_gap and new_next_gap >= next_gap:
                        # Try this replacement
                        candidate = list(seq)
                        candidate[pos] = new_val
                        candidate = try_extend(candidate, deadline - 4)
                        if len(candidate) > len(best_seq):
                            best_seq = candidate
                            seq = candidate
                            improved = True
                        break
                if improved:
                    break
            
            # Also try dropping the last element and re-extending differently
            if not improved and len(seq) >= 3 and time.time() < deadline - 4:
                # Try removing last element and extending with different gap
                trimmed = seq[:-1]
                if len(trimmed) >= 2:
                    last_gap = trimmed[-1] - trimmed[-2]
                    last_val = trimmed[-1]
                    # Try primes starting from last_val + last_gap
                    idx = bisect_left(primes, last_val + last_gap)
                    for try_idx in range(idx, min(idx + 5, P)):
                        candidate = trimmed + [primes[try_idx]]
                        candidate = try_extend(candidate, deadline - 4)
                        if len(candidate) > len(best_seq):
                            best_seq = candidate
                            seq = candidate
                            improved = True
                            break
        seq = best_seq

    return seq
