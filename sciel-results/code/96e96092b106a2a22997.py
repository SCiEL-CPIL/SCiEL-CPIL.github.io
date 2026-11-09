"""Rewritten program: meet-in-the-middle exact search for minimal strict deficit."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np
    from fractions import Fraction

    deadline = time.time() + 85.0  # safety margin under 90s

    denominators = list(range(2, 49))
    n = len(denominators)  # 47

    # Precompute reciprocals as Fractions for exact arithmetic
    # and as floats for fast comparison
    recip_float = np.array([1.0 / d for d in denominators])
    recip_frac = [Fraction(1, d) for d in denominators]

    # Split into two halves
    mid = n // 2  # 23
    left_n = mid
    right_n = n - mid

    # Enumerate all subsets of left half
    # For each subset, record (sum_float, sum_frac, mask)
    # We need sum < 1 strictly.

    # Build left subsets
    left_sums_float = []
    left_sums_frac = []
    left_masks = []

    # Use iterative approach: for each mask 0..2^left_n - 1
    # But 2^23 = 8.4M which is manageable with numpy tricks
    # Actually let's use a smarter approach: build incrementally

    # For left half (indices 0..mid-1), enumerate all 2^mid subsets
    # For right half (indices mid..n-1), enumerate all 2^(n-mid) subsets

    # Strategy: enumerate right subsets, for each right subset with sum_r < 1,
    # find the left subset that maximizes sum_l such that sum_l + sum_r < 1.
    # This is equivalent to: for each right subset, find max sum_l < 1 - sum_r.

    # Better: enumerate all left subsets, sort by sum. For each right subset,
    # binary search for the best left subset.

    # But we also need the combined set to be non-empty (at least one index).
    # And we want to MINIMIZE deficit = 1 - (sum_l + sum_r), i.e., MAXIMIZE sum_l + sum_r subject to sum < 1.

    # Let's enumerate left subsets first.
    # 2^23 = 8,388,608 entries. Each has a float sum and a mask.
    # We can use numpy to generate these efficiently.

    # Generate all subset sums for left half using numpy
    # For each element, either include or not.
    # We can do this with a loop over elements, doubling the array each time.

    # Start with just the empty set
    sums = np.array([0.0])
    masks = np.array([0])

    for i in range(left_n):
        r = recip_float[i]
        # New entries: existing + r, with bit i set
        new_sums = sums + r
        new_masks = masks | (1 << i)
        sums = np.concatenate([sums, new_sums])
        masks = np.concatenate([masks, new_masks])

    # Now sums has 2^left_n entries
    # Sort by sum value
    sort_idx = np.argsort(sums)
    sums_sorted = sums[sort_idx]
    masks_sorted = masks[sort_idx]

    # For right half, enumerate subsets and for each, binary search in left
    # We want max sum_l such that sum_l < 1 - sum_r (strictly)
    # i.e., sum_l <= 1 - sum_r - epsilon, but since we use floats, we need care.
    # Actually we need sum_l + sum_r < 1 strictly. So sum_l < 1 - sum_r.
    # Using np.searchsorted with side='right' gives the first element >= target,
    # so the element before that is the largest < target.

    best_deficit = None  # we want to minimize 1 - total
    best_total = 0.0
    best_left_mask = 0
    best_right_mask = 0

    # Enumerate right subsets
    right_recips = recip_float[mid:]
    right_n_count = right_n

    # Generate right subset sums similarly
    r_sums = np.array([0.0])
    r_masks = np.array([0])

    for i in range(right_n_count):
        r = right_recips[i]
        new_sums = r_sums + r
        new_masks = r_masks | (1 << i)
        r_sums = np.concatenate([r_sums, new_sums])
        r_masks = np.concatenate([r_masks, new_masks])

    # For each right subset, find best left subset
    # We need sum_l < 1 - sum_r, so target = 1.0 - r_sum
    # Use searchsorted: find rightmost index where sums_sorted < target
    # np.searchsorted(sums_sorted, target, side='right') gives first index >= target
    # So the candidate is at index (pos - 1)

    # But we also need to ensure the combined set is non-empty.
    # If both left_mask and right_mask are 0, the set is empty - skip.

    # Process in batches for speed
    # Filter out right subsets where r_sum >= 1 (can't combine with anything to get < 1)
    valid_right = r_sums < 1.0
    r_sums_valid = r_sums[valid_right]
    r_masks_valid = r_masks[valid_right]

    # For each valid right subset, find best left
    targets = 1.0 - r_sums_valid
    # Find insertion point: first index in sums_sorted where value >= target
    positions = np.searchsorted(sums_sorted, targets, side='right')
    # Candidate index is positions - 1
    candidate_idx = positions - 1
    # Filter out candidates that are -1 (no valid left subset)
    valid_cand = candidate_idx >= 0
    candidate_idx = candidate_idx[valid_cand]
    r_sums_valid = r_sums_valid[valid_cand]
    r_masks_valid = r_masks_valid[valid_cand]

    # Get the corresponding left sums and masks
    left_sums_cand = sums_sorted[candidate_idx]
    left_masks_cand = masks_sorted[candidate_idx]

    # Combined sum
    combined_sums = left_sums_cand + r_sums_valid

    # Filter: combined sum must be < 1 (should be by construction, but float errors)
    # Also need non-empty: left_mask != 0 or right_mask != 0
    non_empty = (left_masks_cand != 0) | (r_masks_valid != 0)
    combined_sums = combined_sums[non_empty]
    left_masks_cand = left_masks_cand[non_empty]
    r_masks_valid = r_masks_valid[non_empty]

    # Find the maximum combined sum (which gives minimum deficit)
    if len(combined_sums) > 0:
        max_idx = np.argmax(combined_sums)
        best_total = combined_sums[max_idx]
        best_left_mask = left_masks_cand[max_idx]
        best_right_mask = r_masks_valid[max_idx]

    # Now verify with exact fractions and get the actual indices
    # Build the set of selected indices
    selected = []
    for i in range(left_n):
        if best_left_mask & (1 << i):
            selected.append(i)
    for i in range(right_n_count):
        if best_right_mask & (1 << i):
            selected.append(mid + i)

    # Verify exact sum is strictly below 1
    exact_sum = Fraction(0)
    for idx in selected:
        exact_sum += recip_frac[idx]

    if exact_sum >= 1:
        # This shouldn't happen, but fallback
        # Try removing the last added element
        if selected:
            selected.pop()
            exact_sum = Fraction(0)
            for idx in selected:
                exact_sum += recip_frac[idx]

    if not selected:
        # Fallback: just pick index 0 (denominator 2)
        selected = [0]

    return selected
# EVOLVE-BLOCK-END
