import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140

    max_m = 10**18

    # Sieve sigma(k) for k up to N using numpy
    N = 10_000_000

    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        if i % 2_000_000 == 0:
            if time.time() > deadline - 25:
                N = i
                break

    # Find M = k * sigma(k) for each k, group by M
    m_to_ks = defaultdict(list)
    for k in range(1, N + 1):
        M = k * sigma[k]
        if M <= max_m:
            m_to_ks[M].append(k)
        if time.time() > deadline - 15:
            break

    # Find M with most k values
    best_m = 0
    best_count = 0
    best_ks = []
    for M, ks in m_to_ks.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_m = M
            best_ks = ks

    if best_count == 0:
        return {'M': 333312, 'k': [336, 372, 434]}

    best_ks = list(set(best_ks))
    sqrt_m = math.isqrt(best_m)
    best_ks = [k for k in best_ks if 1 <= k <= sqrt_m]
    best_ks.sort()

    if not best_ks:
        return {'M': 333312, 'k': [336, 372, 434]}

    return {'M': int(best_m), 'k': [int(k) for k in best_ks]}
