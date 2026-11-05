import time
from math import gcd
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    n = 30

    # 1. Compute L = lcm(1, ..., 30)
    def lcm(a, b):
        return a * b // gcd(a, b)

    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)

    # 2. Compute weights w_i = L // i for i = 1..n
    w = [0] * (n + 1)
    for i in range(1, n + 1):
        w[i] = L // i

    # 3. Meet-in-the-middle: split indices 1..30 into two halves
    mid = n // 2  # 15
    left_indices = list(range(1, mid + 1))
    right_indices = list(range(mid + 1, n + 1))

    left_w = [w[i] for i in left_indices]
    right_w = [w[i] for i in right_indices]

    num_left = len(left_w)
    num_right = len(right_w)

    # Generate all left-half sums using iterative base-3 enumeration
    # Store: sum_value -> list of masks (but we only need one mask per sum)
    # For efficiency, use a dict mapping sum to mask

    left_sums_dict = {}
    # Iterate through all 3^15 combinations
    # Use a list to track current digits
    digits = [0] * num_left  # 0=-1, 1=0, 2=+1

    def compute_sum(d, weights):
        s = 0
        for i, d_i in enumerate(d):
            if d_i == 0:
                s -= weights[i]
            elif d_i == 2:
                s += weights[i]
        return s

    # Recursive generation to avoid storing all masks at once
    # But 3^15 = 14M is too large for recursion depth issues, so use iterative

    # Better approach: generate sums iteratively and store in sorted structure
    # Use numpy for vectorized operations in chunks

    # Actually, let's use a smarter approach: generate all left sums, sort them,
    # then for each right sum, binary search for the best match

    # Generate left sums
    left_sums = np.empty(3 ** num_left, dtype=np.int64)
    idx = 0
    # Use iterative base-3 counting
    mask = 0
    for mask in range(3 ** num_left):
        s = 0
        m = mask
        for j in range(num_left):
            digit = m % 3
            m //= 3
            if digit == 0:
                s -= left_w[j]
            elif digit == 2:
                s += left_w[j]
        left_sums[idx] = s
        idx += 1

        if time.time() - start_time > deadline - 30:
            break

    # Sort left sums and keep track of original indices (masks)
    sort_idx = np.argsort(left_sums)
    sorted_sums = left_sums[sort_idx]
    sorted_masks = sort_idx  # The index in left_sums corresponds to the mask

    # Now generate right sums and find best matches
    num_right = len(right_w)
    total_right = 3 ** num_right

    best_abs = np.iinfo(np.int64).max
    best_left_mask = -1
    best_right_mask = -1

    # Process right combinations in batches
    batch_size = 50000
    for batch_start in range(0, total_right, batch_size):
        batch_end = min(batch_start + batch_size, total_right)
        batch_count = batch_end - batch_start

        if time.time() - start_time > deadline:
            break

        # Generate right sums for this batch
        right_sums = np.empty(batch_count, dtype=np.int64)
        for i in range(batch_count):
            mask = batch_start + i
            s = 0
            m = mask
            for j in range(num_right):
                digit = m % 3
                m //= 3
                if digit == 0:
                    s -= right_w[j]
                elif digit == 2:
                    s += right_w[j]
            right_sums[i] = s

        # For each right sum, find best left sum
        targets = -right_sums
        insert_pts = np.searchsorted(sorted_sums, targets, side='left')

        # Check candidate: insert_pts (first >= target)
        valid_a = (insert_pts >= 0) & (insert_pts < len(sorted_sums))
        if np.any(valid_a):
            idx_a = insert_pts[valid_a]
            sums_a = sorted_sums[idx_a]
            totals_a = right_sums[valid_a] + sums_a
            nonzero_a = totals_a != 0
            if np.any(nonzero_a):
                abs_a = np.abs(totals_a[nonzero_a])
                min_pos = np.argmin(abs_a)
                min_val = abs_a[min_pos]
                if min_val < best_abs:
                    valid_idx_in_batch = np.where(valid_a)[0]
                    nonzero_positions = np.where(nonzero_a)[0]
                    actual_batch_idx = valid_idx_in_batch[nonzero_positions[min_pos]]
                    actual_sorted_idx = idx_a[nonzero_a][min_pos]
                    best_abs = min_val
                    best_left_mask = sorted_masks[actual_sorted_idx]
                    best_right_mask = batch_start + actual_batch_idx

        # Check candidate: insert_pts - 1 (last < target)
        valid_b = (insert_pts > 0) & (insert_pts - 1 < len(sorted_sums))
        if np.any(valid_b):
            idx_b = insert_pts[valid_b] - 1
            sums_b = sorted_sums[idx_b]
            totals_b = right_sums[valid_b] + sums_b
            nonzero_b = totals_b != 0
            if np.any(nonzero_b):
                abs_b = np.abs(totals_b[nonzero_b])
                min_pos = np.argmin(abs_b)
                min_val = abs_b[min_pos]
                if min_val < best_abs:
                    valid_idx_in_batch = np.where(valid_b)[0]
                    nonzero_positions = np.where(nonzero_b)[0]
                    actual_batch_idx = valid_idx_in_batch[nonzero_positions[min_pos]]
                    actual_sorted_idx = idx_b[nonzero_b][min_pos]
                    best_abs = min_val
                    best_left_mask = sorted_masks[actual_sorted_idx]
                    best_right_mask = batch_start + actual_batch_idx

    # Reconstruct coefficients from best masks
    result = [0] * n

    if best_left_mask >= 0 and best_right_mask >= 0:
        # Reconstruct left coefficients
        temp = best_left_mask
        for j in range(num_left):
            digit = temp % 3
            temp //= 3
            if digit == 0:
                val = -1
            elif digit == 1:
                val = 0
            else:
                val = 1
            result[left_indices[j] - 1] = val

        # Reconstruct right coefficients
        temp = best_right_mask
        for j in range(num_right):
            digit = temp % 3
            temp //= 3
            if digit == 0:
                val = -1
            elif digit == 1:
                val = 0
            else:
                val = 1
            result[right_indices[j] - 1] = val

        # Local search refinement: try flipping individual coefficients
        # Compute current sum
        current_sum = sum(result[i] / (i + 1) for i in range(n))

        # Try flipping each coefficient and see if it improves
        improved = True
        while improved:
            improved = False
            for i in range(n):
                old_val = result[i]
                # Try other values
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    result[i] = new_val
                    new_sum = sum(result[j] / (j + 1) for j in range(n))
                    if new_sum != 0 and abs(new_sum) < abs(current_sum) * 0.999:
                        current_sum = new_sum
                        improved = True
                        break
                    else:
                        result[i] = old_val

        return result

    # Fallback: return a simple nonzero solution
    fallback = [0] * n
    fallback[0] = 1
    return fallback
