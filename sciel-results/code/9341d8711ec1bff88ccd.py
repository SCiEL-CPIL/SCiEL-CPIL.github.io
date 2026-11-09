"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import bisect
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0  # 5s safety margin

    def time_exceeded():
        return time.time() >= DEADLINE

    def get_primes(limit):
        if limit < 2:
            return []
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                step = i
                start = i * i
                sieve[start::step] = bytearray(len(sieve[start::step]))
        return [i for i in range(2, limit + 1) if sieve[i]]

    # Generate powerful numbers up to L
    L = 50 * 10**6
    primes = get_primes(int(L**0.5) + 1)

    def generate_powerful_iterative():
        pn_set = {1}
        stack = [(0, 1)]
        while stack:
            idx, val = stack.pop()
            if val > L:
                continue
            if val > 1:
                pn_set.add(val)
            for i in range(idx, len(primes)):
                p = primes[i]
                nv = val * p * p
                if nv > L:
                    break
                stack.append((i + 1, nv))
                while nv * p <= L:
                    nv *= p
                    pn_set.add(nv)
                    stack.append((i + 1, nv))
                    if time_exceeded():
                        return pn_set
        return pn_set

    powerful_set = generate_powerful_iterative()

    if time_exceeded():
        return {'N': 2, 'pairs': [[1, 1]]}

    pn_list = sorted(powerful_set)
    n_pn = len(pn_list)

    # Limit K to keep memory and time manageable
    # K^2/2 sums * 8 bytes = memory. For K=8000: 32M*8=256MB, ok.
    K = min(n_pn, 8000)
    pn_arr = np.array(pn_list[:K], dtype=np.int64)

    # Collect all pair sums (a, b) with a <= b
    n_pairs = K * (K + 1) // 2
    all_sums = np.empty(n_pairs, dtype=np.int64)
    pos = 0
    for i in range(K):
        if time_exceeded():
            break
        a = pn_arr[i]
        count = K - i
        all_sums[pos:pos+count] = a + pn_arr[i:]
        pos += count

    if pos == 0:
        return {'N': 2, 'pairs': [[1, 1]]}

    all_sums = all_sums[:pos]

    # Find the most common sum using np.unique
    unique_sums, counts = np.unique(all_sums, return_counts=True)

    # Find the sum with max count
    best_idx = np.argmax(counts)
    best_N = int(unique_sums[best_idx])
    best_count = int(counts[best_idx])

    # Reconstruct pairs for best_N
    best_pairs = []
    for i in range(K):
        a = int(pn_arr[i])
        b = best_N - a
        if b < a:
            break
        j = bisect.bisect_left(pn_arr, b, i)
        if j < K and pn_arr[j] == b:
            best_pairs.append([a, int(b)])

    if not best_pairs:
        return {'N': 2, 'pairs': [[1, 1]]}

    if best_N < 2:
        return {'N': 2, 'pairs': [[1, 1]]}

    return {'N': best_N, 'pairs': best_pairs}

# EVOLVE-BLOCK-END
