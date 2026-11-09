<<<<<<< SEARCH
import time
import bisect
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM
    L = 1
    for d in denominators:
        L = lcm(L, d)
    
    weights = [L // d for d in denominators]
    
    # Split into two halves
    half = n // 2
    left_size = half
    right_size = n - half
    
    left_weights = weights[:left_size]
    right_weights = weights[left_size:]
    
    # Function to generate all subset sums and masks
    def generate_sums(weights_list):
        size = len(weights_list)
        total_masks = 1 << size
        sums = [0] * total_masks
        for i in range(size):
            bit = 1 << i
            w = weights_list[i]
            for mask in range(bit):
                sums[bit | mask] = sums[mask] + w
        
        valid_pairs = [(sums[mask], mask) for mask in range(total_masks) if sums[mask] < L]
        return valid_pairs

    left_pairs = generate_sums(left_weights)
    
    if time.time() > deadline:
        return [0]
    
    right_pairs = generate_sums(right_weights)
    
    if time.time() > deadline:
        return [0]
    
    # Sort left by sum DESCENDING to check best candidates first
    left_pairs.sort(key=lambda x: x[0], reverse=True)
    
    # Sort right by sum ASCENDING for two-pointer
    right_pairs.sort(key=lambda x: x[0])
    
    # Extract sorted right sums and masks
    right_sums = [p[0] for p in right_pairs]
    right_masks = [p[1] for p in right_pairs]
    
    if time.time() > deadline:
        return [0]
    
    # Two-pointer search
    best_deficit = L
    best_left_mask = -1
    best_right_mask = -1
    
    # j points to the largest valid index in right_sums
    j = -1
    n_right = len(right_sums)
    
    for ls, lm in left_pairs:
        if time.time() > deadline:
            break
        
        target = L - ls
        
        # Move j forward while the next right sum is still < target
        # Since ls is decreasing, target is increasing, so j only moves forward
        while j + 1 < n_right and right_sums[j + 1] < target:
            j += 1
            
        if j >= 0:
            rs = right_sums[j]
            rm = right_masks[j]
            total = ls + rs
            deficit = L - total
            if deficit < best_deficit:
                best_deficit = deficit
                best_left_mask = lm
                best_right_mask = rm
                if deficit == 1:
                    break
    
    # Fallback
    if best_left_mask == -1:
        best_left_mask = 0
        best_right_mask = 0
        best_deficit = L

    # Reconstruct indices
    result_indices = []
    for k in range(left_size):
        if best_left_mask & (1 << k):
            result_indices.append(k)
            
    for k in range(right_size):
        if best_right_mask & (1 << k):
            result_indices.append(left_size + k)
            
    # Ensure non-empty
    if not result_indices:
        result_indices = [0]
        
    return result_indices
=======
import time
import bisect
import numpy as np
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    denominators = list(range(2, 49))
    n = len(denominators)

    L = 1
    for d in denominators:
        L = lcm(L, d)

    weights = np.array([L // d for d in denominators], dtype=np.int64)

    half = n // 2
    left_size = half
    right_size = n - half

    left_weights = weights[:left_size]
    right_weights = weights[left_size:]

    # Generate all subset sums for left half using iterative doubling
    total_left = 1 << left_size
    left_sums = np.zeros(total_left, dtype=np.int64)
    for i in range(left_size):
        bit = 1 << i
        w = left_weights[i]
        left_sums[bit:bit*2] = left_sums[:bit] + w

    left_sums = left_sums[left_sums < L]

    if time.time() > deadline:
        return [0]

    # Generate all subset sums for right half
    total_right = 1 << right_size
    right_sums = np.zeros(total_right, dtype=np.int64)
    for i in range(right_size):
        bit = 1 << i
        w = right_weights[i]
        right_sums[bit:bit*2] = right_sums[:bit] + w

    right_sums = right_sums[right_sums < L]

    if time.time() > deadline:
        return [0]

    right_sums.sort()

    # Vectorized search: for each left sum, find max right sum with left+right < L
    targets = L - left_sums
    idxs = np.searchsorted(right_sums, targets, side='left')
    valid = idxs > 0
    if not np.any(valid):
        return [0]

    best_idxs = idxs[valid] - 1
    best_right_vals = right_sums[best_idxs]
    best_left_vals = left_sums[valid]
    deficits = L - best_left_vals - best_right_vals

    min_idx = np.argmin(deficits)
    best_deficit = deficits[min_idx]
    best_ls = best_left_vals[min_idx]
    best_rs = best_right_vals[min_idx]

    if time.time() > deadline:
        return [0]

    # Reconstruct left subset using meet-in-the-middle (12+12)
    mid = left_size // 2
    lw1 = left_weights[:mid]
    lw2 = left_weights[mid:]

    t1 = 1 << mid
    sums1 = np.zeros(t1, dtype=np.int64)
    for i in range(mid):
        bit = 1 << i
        w = lw1[i]
        sums1[bit:bit*2] = sums1[:bit] + w

    t2 = 1 << (left_size - mid)
    sums2 = np.zeros(t2, dtype=np.int64)
    for i in range(left_size - mid):
        bit = 1 << i
        w = lw2[i]
        sums2[bit:bit*2] = sums2[:bit] + w

    sums2_sorted_idx = np.argsort(sums2)
    sums2_sorted = sums2[sums2_sorted_idx]

    left_mask = 0
    found = False
    for i in range(t1):
        s1 = sums1[i]
        s2_needed = best_ls - s1
        if s2_needed < 0:
            continue
        pos = np.searchsorted(sums2_sorted, s2_needed)
        if pos < t2 and sums2_sorted[pos] == s2_needed:
            j = sums2_sorted_idx[pos]
            left_mask = i
            for k in range(mid, left_size):
                if (j >> (k - mid)) & 1:
                    left_mask |= (1 << k)
            found = True
            break

    if not found:
        for k in range(left_size):
            if left_weights[k] <= best_ls:
                left_mask = (1 << k)
                best_ls = left_weights[k]
                target = L - best_ls
                pos = np.searchsorted(right_sums, target, side='left')
                best_rs = right_sums[pos - 1] if pos > 0 else 0
                break

    if time.time() > deadline:
        return [0]

    # Reconstruct right subset using meet-in-the-middle (12+11)
    mid_r = right_size // 2
    rw1 = right_weights[:mid_r]
    rw2 = right_weights[mid_r:]

    t1r = 1 << mid_r
    sums1r = np.zeros(t1r, dtype=np.int64)
    for i in range(mid_r):
        bit = 1 << i
        w = rw1[i]
        sums1r[bit:bit*2] = sums1r[:bit] + w

    t2r = 1 << (right_size - mid_r)
    sums2r = np.zeros(t2r, dtype=np.int64)
    for i in range(right_size - mid_r):
        bit = 1 << i
        w = rw2[i]
        sums2r[bit:bit*2] = sums2r[:bit] + w

    sums2r_sorted_idx = np.argsort(sums2r)
    sums2r_sorted = sums2r[sums2r_sorted_idx]

    right_mask = 0
    found_r = False
    for i in range(t1r):
        s1r = sums1r[i]
        s2r_needed = best_rs - s1r
        if s2r_needed < 0:
            continue
        pos = np.searchsorted(sums2r_sorted, s2r_needed)
        if pos < t2r and sums2r_sorted[pos] == s2r_needed:
            j = sums2r_sorted_idx[pos]
            right_mask = i
            for k in range(mid_r, right_size):
                if (j >> (k - mid_r)) & 1:
                    right_mask |= (1 << k)
            found_r = True
            break

    if not found_r:
        for k in range(right_size):
            if right_weights[k] <= best_rs:
                right_mask = (1 << k)
                break

    result_indices = []
    for k in range(left_size):
        if (left_mask >> k) & 1:
            result_indices.append(k)

    for k in range(right_size):
        if (right_mask >> k) & 1:
            result_indices.append(left_size + k)

    if not result_indices:
        result_indices = [0]

    return result_indices
>>>>>>> REPLACE
