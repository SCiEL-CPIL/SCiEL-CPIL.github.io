"""Covering design C(19,7,4) — vectorized greedy with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools


def construct(v, k, t):
    deadline = time.time() + 115.0  # safety margin before 120s kill
    start = time.time()

    # Precompute all t-subsets as integer masks
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_masks = np.array([sum(1 << x for x in c) for c in all_tsubs], dtype=np.int64)

    # For each point x, precompute which t-subsets contain x (as a boolean mask)
    point_to_tsub = np.zeros((v, n_tsubs), dtype=bool)
    for x in range(v):
        for i, c in enumerate(all_tsubs):
            if x in c:
                point_to_tsub[x, i] = True

    # uncovered: boolean array, True means still uncovered
    uncovered = np.ones(n_tsubs, dtype=bool)
    blocks = []

    while uncovered.any():
        if time.time() > deadline:
            break

        # Pick the first uncovered t-subset as seed
        first_idx = np.argmax(uncovered)
        seed_points = all_tsubs[first_idx]

        block = list(seed_points)

        while len(block) < k:
            if time.time() > deadline:
                break

            # For each candidate point not in block, compute how many uncovered
            # t-subsets it would cover (that also contain all t-1 other points in block)
            # A t-subset is covered by adding x to block iff:
            #   - x is in the t-subset
            #   - all other t-1 points of the t-subset are in block
            #   - the t-subset is currently uncovered

            # Build the set of points already in block
            in_block = np.zeros(v, dtype=bool)
            for p in block:
                in_block[p] = True

            best_gain = -1
            best_point = -1

            # For each candidate point
            for x in range(v):
                if in_block[x]:
                    continue
                # t-subsets containing x: point_to_tsub[x]
                # We need t-subsets where the other t-1 elements are all in block
                # i.e., the t-subset mask & ~x_mask has all bits in block
                # Equivalent: for each t-subset containing x, check if (mask without x) subset of block

                # Vectorized: for t-subsets containing x, check if all other t-1 points are in block
                # A t-subset {a1,...,at} containing x: the other t-1 points must all be in block
                # This is equivalent to: mask & ~(1<<x) has all its set bits in block_mask
                x_mask = 1 << x
                # t-subsets containing x
                contains_x = point_to_tsub[x]
                if not contains_x.any():
                    continue
                # For those t-subsets, check if the remaining t-1 bits are in block
                # remaining_mask = tsub_mask & ~x_mask
                # We need: remaining_mask is a subset of block_mask
                block_mask_val = sum(1 << p for p in block)
                # remaining = tsub_masks & ~x_mask  (element-wise)
                remaining = tsub_masks[contains_x] & ~x_mask
                # Check subset: remaining & ~block_mask_val == 0
                is_subset = (remaining & ~block_mask_val) == 0
                # Also must be uncovered
                gain_count = np.sum(is_subset & uncovered[contains_x])
                if gain_count > best_gain:
                    best_gain = gain_count
                    best_point = x

            if best_point < 0 or best_gain <= 0:
                # No point improves; break to avoid infinite loop
                break

            block.append(best_point)

        if len(block) < k:
            # Pad with arbitrary points not in block to reach size k
            for x in range(v):
                if len(block) >= k:
                    break
                if x not in block:
                    block.append(x)

        block.sort()

        # Check if this block is already in blocks (shouldn't happen but safety)
        block_tuple = tuple(block)
        if block_tuple not in [tuple(b) for b in blocks]:
            blocks.append(block)

        # Update uncovered
        block_mask = sum(1 << p for p in block)
        # A t-subset is covered if all its points are in block
        for i, c in enumerate(all_tsubs):
            if uncovered[i]:
                m = tsub_masks[i]
                if (m & ~block_mask) == 0:
                    uncovered[i] = False

    # If we timed out, we may have uncovered t-subsets. Try a final quick pass
    # to cover remaining with additional blocks if time permits.
    while uncovered.any() and time.time() < deadline - 2.0:
        first_idx = np.argmax(uncovered)
        seed_points = all_tsubs[first_idx]
        block = list(seed_points)
        while len(block) < k:
            in_block = set(block)
            best_gain = -1
            best_point = -1
            for x in range(v):
                if x in in_block:
                    continue
                x_mask = 1 << x
                contains_x = point_to_tsub[x]
                if not contains_x.any():
                    continue
                remaining = tsub_masks[contains_x] & ~x_mask
                block_mask_val = sum(1 << p for p in block)
                is_subset = (remaining & ~block_mask_val) == 0
                gain_count = np.sum(is_subset & uncovered[contains_x])
                if gain_count > best_gain:
                    best_gain = gain_count
                    best_point = x
            if best_point < 0 or best_gain <= 0:
                break
            block.append(best_point)
        if len(block) < k:
            for x in range(v):
                if len(block) >= k:
                    break
                if x not in block:
                    block.append(x)
        block.sort()
        block_tuple = tuple(block)
        if block_tuple not in [tuple(b) for b in blocks]:
            blocks.append(block)
            block_mask = sum(1 << p for p in block)
            for i, c in enumerate(all_tsubs):
                if uncovered[i]:
                    m = tsub_masks[i]
                    if (m & ~block_mask) == 0:
                        uncovered[i] = False

    return blocks
# EVOLVE-BLOCK-END
