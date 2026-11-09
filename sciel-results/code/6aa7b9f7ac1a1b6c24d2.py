import time
import numpy as np
import bisect

def construct():
    t0 = time.time()
    deadline = t0 + 80.0          # safety margin before 90 s kill
    N = 10_000_000

    # ---------- 1. Sieve sigma ----------
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        if time.time() > deadline - 25:
            break
        sigma[d::d] += d

    if time.time() > deadline - 20:
        return [11, 6]            # sigma(6)=sigma(11)=12

    # ---------- 2. Group indices by sigma value (sorted within each group) ----------
    # sigma[0] is unused; we sort indices 1..N by sigma value.
    idx = np.arange(1, N + 1, dtype=np.int64)
    order = np.argsort(sigma[idx], kind='stable')   # stable keeps idx order within ties
    sorted_idx = idx[order]                          # indices grouped by sigma, sorted
    sorted_sig = sigma[sorted_idx]                   # corresponding sigma values (non-decreasing)

    # Build (start, length) for each distinct sigma value.
    # Use np.diff to find where the sigma value changes.
    change = np.empty(len(sorted_sig), dtype=bool)
    change[0] = True
    change[1:] = sorted_sig[1:] != sorted_sig[:-1]
    starts = np.flatnonzero(change)                  # start positions of each group
    ends = np.concatenate([starts[1:], [len(sorted_sig)]])
    n_groups = len(starts)

    # ---------- 3. Search for best (a, b) ----------
    best_num = None   # |2a - 3b|
    best_den = None   # 2b
    best_a = None
    best_b = None

    for g in range(n_groups):
        if time.time() > deadline - 2:
            break
        s0, s1 = starts[g], ends[g]
        length = s1 - s0
        if length < 2:
            continue

        # nums is sorted ascending (because idx was ascending and sort is stable)
        nums = sorted_idx[s0:s1]

        # For each b in nums, find a in nums closest to 1.5*b.
        # Iterate b over the whole group; binary search within the same group.
        for bi in range(length):
            b = int(nums[bi])
            # target = 1.5 * b; we want a minimizing |a - 1.5b| = |2a - 3b|/2
            # Equivalently minimize |2a - 3b| over a in nums.
            # 2a - 3b = 0 => a = 1.5b. Binary search on a for 1.5b.
            # Use integer arithmetic: find a closest to 3b/2.
            # pos = bisect_left(nums, 3*b/2)  -- but 3*b/2 may be non-integer.
            # Better: find a minimizing |2a - 3b|. Since nums is sorted,
            # 2a - 3b is increasing in a. Find a where 2a crosses 3b.
            # i.e., find a such that 2a >= 3b  (first a with a >= ceil(3b/2)).
            # Check that a and the one before.
            # ceil(3b/2) = (3b + 1) // 2
            target_a = (3 * b + 1) // 2   # smallest a with 2a >= 3b
            pos = bisect.bisect_left(nums, target_a)

            # Check pos and pos-1 (and a couple neighbours for safety)
            for p in (pos, pos - 1, pos + 1, pos - 2):
                if p < 0 or p >= length:
                    continue
                a = int(nums[p])
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue   # exact ratio 3/2 -> invalid
                den = 2 * b
                if best_num is None or num * best_den < best_num * den:
                    best_num = num
                    best_den = den
                    best_a = a
                    best_b = b

    if best_a is not None:
        return [int(best_a), int(best_b)]
    return [11, 6]
