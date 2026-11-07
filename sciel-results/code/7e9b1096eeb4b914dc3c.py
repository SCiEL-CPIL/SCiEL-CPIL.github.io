"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s

    N = 10_000_000

    # Step 1: Smallest prime factor sieve
    spf = np.zeros(N + 1, dtype=np.int32)
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i::i] = i

    if time.time() > deadline:
        return [11, 6]

    # Step 2: Compute sigma(n) via multiplicative DP
    sigma = np.ones(N + 1, dtype=np.int64)
    sigma[0] = 0
    p_power = np.zeros(N + 1, dtype=np.int64)

    for n in range(2, N + 1):
        p = spf[n]
        m = n // p
        if m % p == 0:
            p_power[n] = p_power[m] * p
            r = m // p_power[m]
            sigma[n] = sigma[m] + sigma[r] * p_power[n]
        else:
            p_power[n] = p
            sigma[n] = sigma[m] * (1 + p)

    if time.time() > deadline:
        return [11, 6]

    # Step 3: Group by sigma, find pair minimizing |a/b - 3/2|
    sig_vals = sigma[1:]
    n_vals = np.arange(1, N + 1, dtype=np.int64)

    order = np.argsort(sig_vals, kind='mergesort')
    sorted_sig = sig_vals[order]
    sorted_n = n_vals[order]

    diff = np.diff(sorted_sig)
    change_points = np.where(diff != 0)[0] + 1
    boundaries = np.concatenate(([0], change_points, [len(sorted_sig)]))

    best_diff = float('inf')
    best_a = 11
    best_b = 6

    for k in range(len(boundaries) - 1):
        if time.time() > deadline:
            break
        lo = boundaries[k]
        hi = boundaries[k + 1]
        if hi - lo < 2:
            continue

        group_n = sorted_n[lo:hi]
        n_group = len(group_n)

        a_ptr = 0
        for pos_b in range(n_group):
            b_val = group_n[pos_b]
            target = 3.0 * b_val / 2.0

            while a_ptr < n_group - 1 and group_n[a_ptr + 1] <= target:
                a_ptr += 1

            for pos_a in (a_ptr - 1, a_ptr):
                if 0 <= pos_a < n_group:
                    a_val = group_n[pos_a]
                    num = abs(2 * a_val - 3 * b_val)
                    if num == 0:
                        continue
                    err = num / (2.0 * b_val)
                    if err < best_diff:
                        best_diff = err
                        best_a = int(a_val)
                        best_b = int(b_val)

    return [best_a, best_b]
# EVOLVE-BLOCK-END
