# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    # Deadline management: leave 15 seconds for final repair and reduction
    deadline = time.time() + 135.0

    # 1. Precomputation
    tsubs = list(itertools.combinations(range(v), t))
    n_tsub = len(tsubs)
    tsub_to_bit = {sub: i for i, sub in enumerate(tsubs)}

    all_k_subsets = list(itertools.combinations(range(v), k))
    n_ksub = len(all_k_subsets)

    # Build block_masks as Python ints (bitmasks)
    block_masks = []
    for comb in all_k_subsets:
        mask = 0
        for sub in itertools.combinations(comb, t):
            mask |= (1 << tsub_to_bit[sub])
        block_masks.append(mask)

    # Full mask
    full_mask = (1 << n_tsub) - 1
    uncovered = full_mask

    # State
    cover_count = [0] * n_tsub
    blocks_selected = []
    selected_set = set()

    # Precompute: for each t-subset, which blocks contain it
    tsub_to_blocks = [[] for _ in range(n_tsub)]
    for i, mask in enumerate(block_masks):
        m = mask
        while m:
            lsb = m & (-m)
            bit_pos = lsb.bit_length() - 1
            tsub_to_blocks[bit_pos].append(i)
            m ^= lsb

    # Block lookup for repair
    block_to_idx = {comb: i for i, comb in enumerate(all_k_subsets)}

    # 2. Greedy Loop with Candidate Restriction
    while uncovered and time.time() < deadline - 15.0:
        # Pick an uncovered t-subset to use as seed
        uncv = uncovered
        lsb_seed = uncv & (-uncv)
        seed_bit = lsb_seed.bit_length() - 1

        # Get candidate blocks containing this t-subset
        candidates = tsub_to_blocks[seed_bit]

        # Among candidates, find the one with maximum gain
        best_gain = 0
        best_idx = -1
        b_masks = block_masks
        for i in candidates:
            if i in selected_set:
                continue
            g = (b_masks[i] & uncv).bit_count()
            if g > best_gain:
                best_gain = g
                best_idx = i

        if best_gain == 0:
            # Fallback: try all blocks
            for i in range(n_ksub):
                if i in selected_set:
                    continue
                g = (b_masks[i] & uncv).bit_count()
                if g > best_gain:
                    best_gain = g
                    best_idx = i

        if best_gain == 0:
            break

        # Add the best block
        blocks_selected.append(best_idx)
        selected_set.add(best_idx)

        # Update cover_count and uncovered
        mask = block_masks[best_idx]
        m = mask
        while m:
            lsb = m & (-m)
            bit_pos = lsb.bit_length() - 1
            cover_count[bit_pos] += 1
            m ^= lsb

        uncovered &= ~mask

    # 3. Repair Phase
    if uncovered and time.time() < deadline - 10.0:
        m = uncovered
        while m:
            if time.time() > deadline - 5.0:
                break
            lsb = m & (-m)
            bit_pos = lsb.bit_length() - 1
            m ^= lsb

            tsub = tsubs[bit_pos]
            have = set(tsub)
            need = k - len(have)

            remaining_points = [p for p in range(v) if p not in have]

            if need > 0:
                extra = random.sample(remaining_points, need)
            else:
                extra = []

            block = sorted(list(have) + extra)
            block_tuple = tuple(block)

            if block_tuple in selected_set:
                continue

            b_idx = block_to_idx.get(block_tuple)
            if b_idx is not None and b_idx not in selected_set:
                blocks_selected.append(b_idx)
                selected_set.add(b_idx)

                mask = block_masks[b_idx]
                m2 = mask
                while m2:
                    lsb2 = m2 & (-m2)
                    bit_pos2 = lsb2.bit_length() - 1
                    cover_count[bit_pos2] += 1
                    m2 ^= lsb2

                uncovered &= ~mask

    # 4. Reduction Phase - Multiple passes to remove redundant blocks
    for _pass in range(5):
        if time.time() > deadline - 3.0:
            break
        changed = True
        while changed and time.time() < deadline - 3.0:
            changed = False
            for i in range(len(blocks_selected) - 1, -1, -1):
                if time.time() > deadline - 3.0:
                    break
                b_idx = blocks_selected[i]

                # Check if redundant
                mask = block_masks[b_idx]
                m = mask
                redundant = True
                while m:
                    lsb = m & (-m)
                    bit_pos = lsb.bit_length() - 1
                    m ^= lsb
                    if cover_count[bit_pos] <= 1:
                        redundant = False
                        break

                if redundant:
                    mask = block_masks[b_idx]
                    m = mask
                    while m:
                        lsb = m & (-m)
                        bit_pos = lsb.bit_length() - 1
                        m ^= lsb
                        cover_count[bit_pos] -= 1

                    blocks_selected.pop(i)
                    selected_set.remove(b_idx)
                    changed = True
                    break

    # 5. Final Conversion
    result = []
    for idx in blocks_selected:
        result.append(list(all_k_subsets[idx]))

    return result

# EVOLVE-BLOCK-END
