import numpy as np
import time

def construct():
    N = 10**7
    t0 = time.time()
    deadline = t0 + 85  # safety margin before 90s kill

    # Step 1: Compute sigma(n) for n = 1..N using vectorized NumPy sieve
    # sigma(n) = sum of divisors
    # For each d from 1 to N, add d to all multiples of d
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d

    # Step 2: Sort indices 1..N by sigma value
    indices = np.arange(1, N + 1)
    sorted_indices = indices[np.argsort(sigma[1:N+1], kind='stable')]
    sorted_sigma = sigma[1:N+1][np.argsort(sigma[1:N+1], kind='stable')]

    # Step 3: For each group (fiber) with same sigma, find pair minimizing |2a - 3b|
    best_err = np.inf
    best_a = 0
    best_b = 0

    i = 0
    n = N
    while i < n:
        j = i + 1
        while j < n and sorted_sigma[j] == sorted_sigma[i]:
            j += 1
        # Group is sorted_indices[i:j]
        if j - i >= 2:
            # Two-pointer: for each b in group, find a closest to 3b/2
            group = sorted_indices[i:j]
            # group is sorted
            ptr = 0
            for k in range(len(group)):
                b = group[k]
                target = 3 * b / 2.0
                # Advance ptr while group[ptr+1] is closer to target
                while ptr < len(group) - 1 and abs(group[ptr + 1] - target) < abs(group[ptr] - target):
                    ptr += 1
                # Check group[ptr] and maybe ptr-1, ptr+1
                for a_candidate in [group[ptr]]:
                    if a_candidate != b:
                        err = abs(2 * a_candidate - 3 * b)
                        if err < best_err:
                            best_err = err
                            best_a = int(a_candidate)
                            best_b = int(b)
        i = j

    if time.time() > deadline:
        if best_a == 0:
            best_a, best_b = 11, 6
        return [int(best_a), int(best_b)]

    if best_a == 0:
        best_a, best_b = 11, 6

    return [int(best_a), int(best_b)]
