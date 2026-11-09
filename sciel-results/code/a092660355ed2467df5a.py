import time
import math
import numpy as np

def construct():
    start_time = time.time()
    MAX_N = 1000000
    MIN_N = 10000
    N_LIMIT = MAX_N + 1
    DEADLINE = 80

    # 1. Compute Smallest Prime Factor (SPF)
    spf = np.arange(N_LIMIT + 1, dtype=np.int64)
    for i in range(2, int(math.isqrt(N_LIMIT)) + 1):
        if spf[i] == i:
            for j in range(i * i, N_LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i

    if time.time() - start_time > DEADLINE:
        return MIN_N

    # 2. Compute powerful part using recurrence with pmax tracking
    # pmax[i] = highest power of spf[i] that divides i
    # pp[i] = powerful part of i
    pp = np.ones(N_LIMIT + 1, dtype=np.int64)
    pmax = np.ones(N_LIMIT + 1, dtype=np.int64)

    for i in range(2, N_LIMIT + 1):
        p = spf[i]
        j = i // p
        if j % p == 0:
            # p divides j, so pmax[j] >= p
            pmax[i] = pmax[j] * p
        else:
            # p does not divide j
            pmax[i] = p

        # pp[i] = pmax[i] * pp[i // pmax[i]] if pmax[i] >= p^2, else pp[i // p]
        if pmax[i] >= p * p:
            pp[i] = pmax[i] * pp[i // pmax[i]]
        else:
            pp[i] = pp[j]

        if time.time() - start_time > DEADLINE - 5:
            break

    # 3. Find max product
    products = pp[MIN_N:MAX_N + 1] * pp[MIN_N + 1:MAX_N + 2]

    if len(products) == 0:
        return MIN_N

    max_idx = np.argmax(products)
    best_n = MIN_N + int(max_idx)

    return int(best_n)
