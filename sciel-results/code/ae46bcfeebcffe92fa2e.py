import time
import math
import bisect
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill

    N = 10_000_000

    # Fast SPF sieve using numpy vectorized operations
    spf = np.zeros(N + 1, dtype=np.int32)
    spf[0] = 0
    spf[1] = 1
    # Initialize with self
    idx = np.arange(2, N + 1)
    spf[idx] = idx

    # Sieve: for each prime p, mark multiples
    limit = int(math.isqrt(N))
    for i in range(2, limit + 1):
        if spf[i] == i:  # i is prime
            # Mark multiples of i that haven't been marked yet
            multiples = np.arange(i * i, N + 1, i)
            mask = spf[multiples] == multiples
            if np.any(mask):
                spf[multiples[mask]] = i

    if time.time() - start_time > deadline - 35:
        return [int(122), int(80)]

    # Compute sigma(n) for all n using multiplicative property
    sigma = np.zeros(N + 1, dtype=np.int64)
    sigma[1] = 1
    spow = np.zeros(N + 1, dtype=np.int64)
    ssigma = np.zeros(N + 1, dtype=np.int64)
    spow[1] = 1
    ssigma[1] = 1

    for n in range(2, N + 1):
        if (n & 0xFFFF) == 0 and time.time() - start_time > deadline - 25:
            break
        p = int(spf[n])
        m = n // p
        if int(spf[m]) != p:
            # p does not divide m
            sigma[n] = sigma[m] * (1 + p)
            spow[n] = p
            ssigma[n] = 1 + p
        else:
            # p divides m
            r_sigma = sigma[m] // ssigma[m]
            new_spow = spow[m] * p
            sigma[n] = sigma[m] + r_sigma * new_spow
            spow[n] = new_spow
            ssigma[n] = ssigma[m] + new_spow

    if time.time() - start_time > deadline - 12:
        return [int(122), int(80)]

    # Group by sigma value - only keep groups with at least 2 elements
    # Use a dictionary mapping sigma -> list of n
    groups = defaultdict(list)
    for n in range(1, N + 1):
        s_val = int(sigma[n])
        if s_val not in groups:
            groups[s_val] = []
        groups[s_val].append(n)

    if time.time() - start_time > deadline - 5:
        return [int(122), int(80)]

    best_error = float('inf')
    best_a = 122
    best_b = 80

    # Process groups: for each group, find pairs (a,b) minimizing |a/b - 3/2|
    # err = |2a - 3b| / (2b), skip if 2a == 3b
    for s_val, nums in groups.items():
        if time.time() - start_time > deadline:
            break
        if len(nums) < 2:
            continue

        nums_sorted = sorted(nums)
        ln = len(nums_sorted)

        # For each b in the group, find a closest to 1.5*b
        # Use two-pointer or binary search
        # Binary search approach: for each b, target_a = 1.5 * b
        for i in range(ln):
            b = nums_sorted[i]
            target = 3 * b // 2  # floor of 1.5*b; we'll check neighbors
            # We want a such that 2a ≈ 3b, i.e., a ≈ 3b/2
            # Check a values around 3b/2
            target_float = 1.5 * b
            idx = bisect.bisect_left(nums_sorted, target_float)
            candidates = []
            if idx < ln:
                candidates.append(nums_sorted[idx])
            if idx > 0:
                candidates.append(nums_sorted[idx - 1])
            # Also check idx+1 and idx-2 for safety
            if idx + 1 < ln:
                candidates.append(nums_sorted[idx + 1])
            if idx - 2 >= 0:
                candidates.append(nums_sorted[idx - 2])

            for a in candidates:
                if a == b:
                    continue
                two_a = 2 * a
                three_b = 3 * b
                if two_a == three_b:
                    continue  # exact ratio 3/2 is invalid
                num = abs(two_a - three_b)
                den = 2 * b
                err = num / den
                if err < best_error:
                    best_error = err
                    best_a = a
                    best_b = b

    if best_error == float('inf'):
        return [int(122), int(80)]
    return [int(best_a), int(best_b)]
