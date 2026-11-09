import numpy as np
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 110.0

    # Sieve primes up to 10^6
    limit = 1000000
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.array([i for i, val in enumerate(is_prime) if val])

    best_k = 1
    best_s = 1

    # Dictionary to track last occurrence of each phi value
    # Maps phi_value -> last global index where this phi was seen
    last_occurrence = {}
    left_idx = 0  # Global index of left pointer of current valid window

    def update_window(idx, val):
        nonlocal best_k, best_s, left_idx

        # Check if this phi value has been seen before
        if val in last_occurrence:
            prev_idx = last_occurrence[val]
            if prev_idx + 1 > left_idx:
                left_idx = prev_idx + 1

        # Update the last occurrence
        last_occurrence[val] = idx

        # Update best if current window is longer
        current_len = idx - left_idx + 1
        if current_len > best_k:
            best_k = current_len
            best_s = left_idx
            if best_k >= 200:
                return True  # Signal to stop

        return False

    def compute_phi_segmented(start_val, end_val):
        """Compute phi for range [start_val, end_val] using segmented sieve."""
        n = end_val - start_val + 1
        phis = np.arange(start_val, end_val + 1, dtype=np.int64)
        remaining = np.arange(start_val, end_val + 1, dtype=np.int64)

        for p in primes:
            if p * p > end_val:
                break
            first_mult = ((start_val + p - 1) // p) * p
            start_idx = first_mult - start_val
            step = p
            mask_indices = np.arange(start_idx, n, step)

            if len(mask_indices) == 0:
                continue

            phi_slice = phis[mask_indices]
            phis[mask_indices] = phi_slice // p * (p - 1)

            # Remove all factors of p from remaining
            rem_slice = remaining[mask_indices].copy()
            while np.any(rem_slice % p == 0):
                rem_slice[rem_slice % p == 0] //= p
            remaining[mask_indices] = rem_slice

        # Handle remaining prime factors > sqrt(end_val)
        mask_rem = remaining > 1
        if np.any(mask_rem):
            phis[mask_rem] = phis[mask_rem] // remaining[mask_rem] * (remaining[mask_rem] - 1)

        return phis

    # Search strategy: start from high ranges where collisions are rarer
    # Use multiple segments, searching from high to low
    segment_size = 500000
    max_val = 10**12

    # First pass: search in high range [10^11, 10^12]
    # Then if needed, search lower ranges
    search_ranges = []
    
    # High range first (most promising)
    high_start = int(10**11)
    high_end = max_val
    for seg_start in range(high_start, high_end + 1, segment_size):
        seg_end = min(seg_start + segment_size - 1, high_end)
        search_ranges.append((seg_start, seg_end))
    
    # Medium range
    med_start = int(10**10)
    med_end = high_start - 1
    for seg_start in range(med_start, med_end + 1, segment_size):
        seg_end = min(seg_start + segment_size - 1, med_end)
        search_ranges.append((seg_start, seg_end))
    
    # Low range (least promising, search last)
    low_start = 1
    low_end = med_start - 1
    for seg_start in range(low_start, low_end + 1, segment_size):
        seg_end = min(seg_start + segment_size - 1, low_end)
        search_ranges.append((seg_start, seg_end))

    # Process segments
    for seg_start, seg_end in search_ranges:
        if time.time() > deadline:
            break
        if best_k >= 200:
            break

        phis = compute_phi_segmented(seg_start, seg_end)
        phis_list = phis.tolist()

        for i, val in enumerate(phis_list):
            idx = seg_start + i
            should_stop = update_window(idx, val)
            if should_stop:
                break

        if best_k >= 200:
            break

        # Clean up old entries from last_occurrence to save memory
        if len(last_occurrence) > 2000000:
            keys_to_remove = [k for k, v in last_occurrence.items() if v < left_idx]
            for k in keys_to_remove:
                del last_occurrence[k]

    # Guarantee valid result
    if best_s + best_k - 1 > 10**12:
        best_s = 1
        best_k = 1

    return {'start': best_s, 'length': best_k}
