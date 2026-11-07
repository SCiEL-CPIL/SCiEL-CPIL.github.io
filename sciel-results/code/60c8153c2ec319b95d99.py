import numpy as np
import time
from bisect import bisect_left

P = {'maximum': 10000000, 'target': [3, 2]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    N = 3000000

    sigma = np.zeros(N + 1, dtype=np.int64)

    half = N // 2
    for i in range(1, half + 1):
        if time.time() > deadline - 30.0:
            return [11, 6]
        sigma[i::i] += i

    sigma[half+1:] += np.arange(half+1, N+1, dtype=np.int64)

    if time.time() > deadline - 20.0:
        return [11, 6]

    # Build groups: map sigma value -> list of n
    groups = {}
    for n in range(1, N + 1):
        s = int(sigma[n])
        if s not in groups:
            groups[s] = []
        groups[s].append(n)

    best_value = 1.0
    best_pair = (11, 6)

    # Process groups in order of size (largest first) to find good pairs quickly
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)

    for s, nums in sorted_groups:
        if len(nums) < 2:
            continue
        if time.time() > deadline - 5.0:
            break
        nums.sort()
        nlen = len(nums)
        for b_idx in range(nlen):
            b = nums[b_idx]
            if b_idx % 200 == 0 and time.time() > deadline - 5.0:
                break
            # We want a/b ≈ 3/2, so a ≈ 3b/2
            target_num = 3 * b
            target = target_num // 2
            # Check a = target and neighbors
            candidates = []
            if target >= 1 and target <= N:
                candidates.append(target)
                if target - 1 >= 1:
                    candidates.append(target - 1)
                if target + 1 <= N:
                    candidates.append(target + 1)
            # Also use bisect to find nearest in the group
            idx = bisect_left(nums, target)
            if idx < nlen:
                candidates.append(nums[idx])
            if idx > 0:
                candidates.append(nums[idx - 1])
            for a in candidates:
                if a == b or a < 1 or a > N:
                    continue
                # Check if sigma(a) == sigma(b) == s
                if int(sigma[a]) != s:
                    continue
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue
                val = num / (2.0 * b)
                if val < best_value:
                    best_value = val
                    best_pair = (a, b)

    return [int(best_pair[0]), int(best_pair[1])]
# EVOLVE-BLOCK-END
