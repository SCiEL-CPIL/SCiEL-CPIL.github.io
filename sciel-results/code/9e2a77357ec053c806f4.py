import numpy as np
import time

def construct():
    t_start = time.time()
    deadline = t_start + 85.0

    N = 10_000_000

    # Step 1: Sieve for smallest prime factor
    spf = np.zeros(N + 1, dtype=np.int64)
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            if i * i <= N:
                for j in range(i * i, N + 1, i):
                    if spf[j] == 0:
                        spf[j] = i

    if time.time() > deadline - 15.0:
        return [1, 1]

    # Step 2: Compute sigma(n) using multiplicative property
    sigma_arr = np.ones(N + 1, dtype=np.int64)
    p_pow = np.ones(N + 1, dtype=np.int64)
    p_sum = np.ones(N + 1, dtype=np.int64)

    for n in range(2, N + 1):
        if time.time() > deadline - 10.0:
            break
        p = spf[n]
        m = n // p
        if m % p == 0:
            p_pow[n] = p_pow[m] * p
            p_sum[n] = p_sum[m] + p_pow[n]
            sigma_arr[n] = sigma_arr[m] * p_sum[n] // p_sum[m]
        else:
            p_pow[n] = p
            p_sum[n] = 1 + p
            sigma_arr[n] = sigma_arr[m] * (1 + p)

    if time.time() > deadline - 5.0:
        return [1, 1]

    # Step 3: Sort by sigma value
    indices = np.arange(1, N + 1)
    keys = sigma_arr[indices]
    order = np.lexsort((indices, keys))
    sorted_n = indices[order]
    sorted_s = keys[order]

    # Step 4: Build range mapping
    sigma_to_range = {}
    current_s = sorted_s[0]
    start_idx = 0
    for i in range(1, len(sorted_s)):
        if sorted_s[i] != current_s:
            sigma_to_range[current_s] = (start_idx, i)
            current_s = sorted_s[i]
            start_idx = i
    sigma_to_range[current_s] = (start_idx, len(sorted_s))

    if time.time() > deadline - 1.0:
        return [1, 1]

    # Step 5: Search for best pair
    best_err = float('inf')
    best_a = 1
    best_b = 1

    for b in range(1, N + 1):
        if time.time() > deadline - 0.5:
            break
        s = sigma_arr[b]
        if s not in sigma_to_range:
            continue
        start, end = sigma_to_range[s]
        if end - start < 2:
            continue
        target = 1.5 * b
        sub_n = sorted_n[start:end]
        pos = np.searchsorted(sub_n, target)
        candidates = []
        if pos > 0:
            candidates.append(sub_n[pos-1])
        if pos < len(sub_n):
            candidates.append(sub_n[pos])
        for a in candidates:
            a_int = int(a)
            if a_int == b:
                continue
            if 2 * a_int == 3 * b:
                continue
            err = abs(a_int / b - 1.5)
            if err < best_err:
                best_err = err
                best_a = a_int
                best_b = b

    return [int(best_a), int(best_b)]
