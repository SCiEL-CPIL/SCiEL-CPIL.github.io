import sys
import time
import math
import numpy as np
from collections import defaultdict
from bisect import bisect_left

MAX_N = 10000000

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    # 1. Compute sigma(n) for all n <= MAX_N using numpy sieve
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)

    for i in range(1, MAX_N + 1):
        if time.time() > deadline - 15:
            return [11, 6]
        sigma[i::i] += i

    if time.time() > deadline:
        return [11, 6]

    # 2. Build groups: sigma_value -> sorted list of n
    sigma_groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        if time.time() > deadline - 5:
            break
        s = int(sigma[n])
        sigma_groups[s].append(n)

    if time.time() > deadline:
        return [11, 6]

    # 3. Search for best pair, processing b in descending order
    best_a, best_b = 11, 6
    best_num, best_den = 4, 12  # |2*11 - 3*6| / (2*6) = 4/12

    for b in range(MAX_N, 0, -1):
        if time.time() > deadline - 2:
            break

        # Early termination: if best error < 1/(2*b), no smaller b can improve
        if best_num * 2 * b < best_den:
            break

        s = int(sigma[b])
        L = sigma_groups.get(s)
        if L is None or len(L) < 2:
            continue

        target = 1.5 * b
        idx = bisect_left(L, target)

        candidates = []
        if idx < len(L):
            candidates.append(L[idx])
        if idx > 0:
            candidates.append(L[idx - 1])

        for a in candidates:
            if a == b:
                continue
            num = abs(2 * a - 3 * b)
            if num == 0:
                continue
            den = 2 * b
            if num * best_den < best_num * den:
                best_a, best_b = a, b
                best_num, best_den = num, den

    return [int(best_a), int(best_b)]
