import time
import numpy as np
from math import isqrt

def _sieve(n):
    is_prime = bytearray(b'\x01') * (n + 1)
    is_prime[0:2] = b'\x00\x00'
    for p in range(2, isqrt(n) + 1):
        if is_prime[p]:
            start = p * p
            is_prime[start:n + 1:p] = b'\x00' * (((n - start) // p) + 1)
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    t0 = time.time()
    DEADLINE = 85.0

    primes = _sieve(10000)
    n = len(primes)

    if n <= 1:
        return primes

    primes_arr = np.array(primes, dtype=np.int64)

    # states[i] = list of (last_gap, length, prev_index)
    # Pareto-optimal: sorted by last_gap ascending, length strictly increasing
    # (0, 1, -1) means sequence of length 1 ending at primes[i]
    states = [[] for _ in range(n)]
    states[0].append((0, 1, -1))

    best_length = 1
    best_end = 0

    for i in range(1, n):
        if time.time() - t0 > DEADLINE:
            break

        candidates = [(0, 1, -1)]  # just primes[i] alone

        pi = primes[i]
        gaps = pi - primes_arr[:i]

        for j in range(i):
            gap = int(gaps[j])
            sj = states[j]
            if not sj:
                continue

            # Binary search: rightmost entry in sj with last_gap <= gap
            lo, hi = 0, len(sj)
            while lo < hi:
                mid = (lo + hi) // 2
                if sj[mid][0] <= gap:
                    lo = mid + 1
                else:
                    hi = mid
            if lo > 0:
                last_gap_j, len_j, prev_j = sj[lo - 1]
                candidates.append((gap, len_j + 1, j))

        # Filter to Pareto-optimal: sort by gap, keep strictly increasing length
        if candidates:
            candidates.sort(key=lambda x: (x[0], -x[1]))
            filtered = []
            max_len = 0
            for gap, length, prev in candidates:
                if length > max_len:
                    filtered.append((gap, length, prev))
                    max_len = length
            states[i] = filtered

            if filtered:
                last_len = filtered[-1][1]
                if last_len > best_length:
                    best_length = last_len
                    best_end = i

    # Reconstruct the sequence ending at best_end
    if best_length <= 1:
        return [primes[best_end]]

    seq = []
    idx = best_end
    while idx != -1:
        seq.append(primes[idx])
        sj = states[idx]
        if not sj:
            break
        _, _, prev = sj[-1]
        idx = prev
    seq.reverse()

    # Local search: try to extend the sequence
    if time.time() - t0 < DEADLINE - 3.0 and len(seq) > 1:
        seq = _local_search(seq, primes, t0, DEADLINE)

    return seq

def _local_search(seq, primes, t0, DEADLINE):
    """Try to improve the sequence by local modifications and extensions."""
    best_seq = list(seq)
    prime_index = {p: i for i, p in enumerate(primes)}
    n_primes = len(primes)

    def is_valid(s):
        if len(s) < 2:
            return True
        gaps = [s[i+1] - s[i] for i in range(len(s)-1)]
        return all(gaps[i+1] >= gaps[i] for i in range(len(gaps)-1))

    # Strategy 1: Greedy extension from the end
    extended = list(best_seq)
    while time.time() - t0 < DEADLINE - 3.0:
        if len(extended) < 2:
            break
        last_gap = extended[-1] - extended[-2]
        last_prime = extended[-1]
        last_idx = prime_index.get(last_prime, -1)
        if last_idx < 0:
            break
        next_prime = None
        for new_idx in range(last_idx + 1, n_primes):
            new_val = primes[new_idx]
            new_gap = new_val - last_prime
            if new_gap >= last_gap:
                next_prime = new_val
                break
        if next_prime is None:
            break
        extended.append(next_prime)

    if len(extended) > len(best_seq):
        best_seq = extended

    # Strategy 2: Try replacing individual elements with nearby primes to enable extension
    if time.time() - t0 < DEADLINE - 3.0:
        for trial in range(20):
            if time.time() - t0 > DEADLINE - 3.0:
                break
            improved = False
            for pos in range(len(best_seq)):
                old_val = best_seq[pos]
                idx = prime_index.get(old_val, -1)
                if idx < 0:
                    continue
                for delta in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
                    new_idx = idx + delta
                    if 0 <= new_idx < n_primes:
                        new_val = primes[new_idx]
                        if new_val == old_val:
                            continue
                        candidate = list(best_seq)
                        candidate[pos] = new_val
                        # Check if still increasing
                        if pos > 0 and candidate[pos] <= candidate[pos-1]:
                            continue
                        if pos < len(candidate) - 1 and candidate[pos] >= candidate[pos+1]:
                            continue
                        if is_valid(candidate):
                            # Try to extend this candidate
                            ext = list(candidate)
                            while time.time() - t0 < DEADLINE - 3.0:
                                if len(ext) < 2:
                                    break
                                lg = ext[-1] - ext[-2]
                                lp = ext[-1]
                                li = prime_index.get(lp, -1)
                                if li < 0:
                                    break
                                found = False
                                for ni in range(li + 1, n_primes):
                                    nv = primes[ni]
                                    if nv - lp >= lg:
                                        ext.append(nv)
                                        found = True
                                        break
                                if not found:
                                    break
                            if len(ext) > len(best_seq):
                                best_seq = ext
                                improved = True
                                break
            if improved:
                break

    return best_seq
