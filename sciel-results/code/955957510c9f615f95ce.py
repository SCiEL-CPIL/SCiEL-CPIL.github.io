import numpy as np
import time
from bisect import bisect_left

P = {'maximum': 10000000, 'target': [3, 2]}

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill

    N = 10000000

    # Compute sigma(n) for all n up to N using sieve
    sigma = np.zeros(N + 1, dtype=np.int64)

    # Sieve for sigma: for each i, add i to all multiples of i
    # Optimization: split into two phases to allow early exit
    half = N // 2
    for i in range(1, half + 1):
        if time.time() > deadline - 30.0:
            return [11, 6]
        sigma[i::i] += i

    # For i > N/2, only the multiple i itself gets i added
    sigma[half+1:] += np.arange(half+1, N+1, dtype=np.int64)

    if time.time() > deadline - 15.0:
        return [11, 6]

    # Group numbers by sigma value
    groups = {}
    for n in range(1, N + 1):
        s = int(sigma[n])
        if s not in groups:
            groups[s] = []
        groups[s].append(n)

    if time.time() > deadline - 5.0:
        return [11, 6]

    # Search for best pair
    best_error = float('inf')
    best_pair = (11, 6)

    # Check seed pair first
    if sigma[11] == sigma[6]:
        val = abs(2*11 - 3*6) / (2.0 * 6)
        if val < best_error:
            best_error = val
            best_pair = (11, 6)

    # Iterate over groups with at least 2 elements
    for s, nums in groups.items():
        if time.time() > deadline - 2.0:
            break
        if len(nums) < 2:
            continue

        nums.sort()
        nlen = len(nums)

        # For each b in nums, find a in nums such that a/b is close to 1.5
        for b_idx in range(nlen):
            if b_idx % 1000 == 0 and time.time() > deadline - 2.0:
                break
            b = nums[b_idx]
            target = 1.5 * b
            idx = bisect_left(nums, target)

            candidates = set()
            if idx < nlen:
                candidates.add(nums[idx])
            if idx > 0:
                candidates.add(nums[idx-1])

            for a in candidates:
                if a == b:
                    continue
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue
                err = num / (2.0 * b)
                if err < best_error:
                    best_error = err
                    best_pair = (a, b)

    return [int(best_pair[0]), int(best_pair[1])]
