import numpy as np
from collections import defaultdict

def construct():
    N = 10_000_000
    # Compute sigma(n) for n = 1..N using sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i

    # Sort indices 1..N by their sigma value
    idx = np.arange(1, N + 1, dtype=np.int64)
    sigma_vals = sigma[1:]
    order = np.argsort(sigma_vals, kind='stable')
    sorted_idx = idx[order]
    sorted_sigma = sigma_vals[order]

    best_val = float('inf')
    best_a = 1
    best_b = 1

    # Traverse sorted array, find groups with same sigma
    i = 0
    total = N
    while i < total:
        j = i + 1
        while j < total and sorted_sigma[j] == sorted_sigma[i]:
            j += 1
        group_size = j - i
        if group_size >= 2:
            # Extract group
            group = sorted_idx[i:j]
            g_len = len(group)
            for a_pos in range(g_len):
                a = int(group[a_pos])
                for b_pos in range(g_len):
                    if a_pos == b_pos:
                        continue
                    b = int(group[b_pos])
                    # |a/b - 3/2| = |2a - 3b| / (2b)
                    num = abs(2 * a - 3 * b)
                    if num == 0:
                        continue
                    val = num / (2.0 * b)
                    if val < best_val:
                        best_val = val
                        best_a = a
                        best_b = b
        i = j

    return [best_a, best_b]
