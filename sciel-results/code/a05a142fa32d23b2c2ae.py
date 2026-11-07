"""Find a,b <= 1e7 with sigma(a)=sigma(b) minimizing |a/b - 3/2|."""
import time
import sys
from collections import defaultdict
import bisect
import numpy as np

def construct():
    start = time.time()
    deadline = start + 80.0
    MAX_N = 10000000

    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    for d in range(1, MAX_N + 1):
        sigma[d::d] += d

    if time.time() > deadline:
        return [11, 6]

    indices = np.arange(1, MAX_N + 1)
    sigma_vals = sigma[1:]
    order = np.argsort(sigma_vals, kind='stable')
    sorted_sigma = sigma_vals[order]
    sorted_indices = indices[order]

    if time.time() > deadline:
        return [11, 6]

    best_a, best_b = 11, 6
    best_num, best_den = 4, 12

    i = 0
    n = MAX_N
    while i < n:
        if time.time() > deadline:
            break
        j = i + 1
        while j < n and sorted_sigma[j] == sorted_sigma[i]:
            j += 1

        if j - i >= 2:
            group = sorted_indices[i:j].tolist()
            for b in group:
                target = 3 * b // 2
                pos = bisect.bisect_left(group, target)
                for idx in [pos - 1, pos]:
                    if 0 <= idx < len(group):
                        a = group[idx]
                        if a == b:
                            continue
                        num = abs(2 * a - 3 * b)
                        if num == 0:
                            continue
                        den = 2 * b
                        if num * best_den < best_num * den:
                            best_a, best_b = a, b
                            best_num, best_den = num, den

        i = j

    if best_num == 0:
        best_a, best_b = 11, 6
        best_num, best_den = 4, 12

    return [int(best_a), int(best_b)]
