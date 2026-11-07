import numpy as np
import time
from bisect import bisect_left

P = {'maximum': 10000000, 'target': [3, 2]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    N = 10000000

    # Compute sigma(n) for all n up to N using sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i

    # For each b, find a near 3b/2 with sigma(a) == sigma(b)
    best_value = float('inf')
    best_pair = (11, 6)

    # Check the seed first
    if sigma[11] == sigma[6]:
        val = abs(2*11 - 3*6) / (2.0 * 6)
        if val < best_value:
            best_value = val
            best_pair = (11, 6)

    for b in range(1, N + 1):
        if time.time() > deadline - 2.0:
            break
        if b % 100000 == 0 and time.time() > deadline - 5.0:
            break

        sb = int(sigma[b])
        target = 3 * b // 2

        # Check candidates around target
        for a in [target - 1, target, target + 1, target - 2, target + 2]:
            if a < 1 or a > N or a == b:
                continue
            if int(sigma[a]) == sb:
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue
                val = num / (2.0 * b)
                if val < best_value:
                    best_value = val
                    best_pair = (a, b)

    return [int(best_pair[0]), int(best_pair[1])]
