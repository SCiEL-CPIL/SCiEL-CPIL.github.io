"""Covering design C(v,k,t) via bitmask greedy + local-search reduction."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np
from math import comb


def _combo_to_mask(combo, v):
    """Convert a tuple of point indices to a bit integer."""
    m = 0
    for p in combo:
        m |= (1 << p)
    return m


def _mask_to_combo(mask, v):
    """Convert a bit integer back to a sorted tuple of point indices."""
    out = []
    i = 0
    while mask:
        if mask & 1:
            out.append(i)
        mask >>= 1
        i += 1
    return tuple(out)


def _gen_tsubsets_of_block(block, t):
    """Yield all t-subsets of a sorted block tuple."""
    return itertools.combinations(block, t)


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    def time_ok():
        return time.time() < deadline

    all_points = list(range(v))
    all_t_subs = list(itertools.combinations(all_points, t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute bitmask for each t-subset (v <= 64 fits in a Python int)
    t_sub_masks = [_combo_to_mask(sub, v) for sub in all_t_subs]

    covered = np.zeros(n_t, dtype=np.uint8)
    uncovered_count = n_t
    blocks = []
    block_masks = []  # bitmask of each chosen block

    # Precompute: for each point, list of t-subset indices that contain it
    # (helps with quick gain estimation via anchor)
    point_to_tsubs = [[] for _ in range(v)]
    for i, sub in enumerate(all_t_subs):
        for p in sub:
            point_to_tsubs[p].append(i)

    # Greedy Phase
    while uncovered_count > 0 and time_ok():
        # Pick a random uncovered t-subset as anchor
        uncovered_indices = np.where(covered == 0)[0]
        if len(uncovered_indices) == 0:
            break

        anchor_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
        anchor_t_sub = all_t_subs[anchor_idx]
        anchor_set = set(anchor_t_sub)

        remaining_points = [p for p in all_points if p not in anchor_set]
        k_minus_t = k - t

        best_block = None
        best_gain = -1
        best_block_mask = 0

        # Iterate all extensions
        for ext in itertools.combinations(remaining_points, k_minus_t):
            block = sorted(list(anchor_t_sub) + list(ext))
            # Compute gain: count uncovered t-subsets contained in this block
            gain = 0
            for t_sub in itertools.combinations(block, t):
                idx = t_sub_to_idx[t_sub]
                if covered[idx] == 0:
                    gain += 1

            if gain > best_gain:
                best_gain = gain
                best_block = block
                best_block_mask = _combo_to_mask(block, v)

        if best_block is None:
            break

        blocks.append(best_block)
        block_masks.append(best_block_mask)

        # Update coverage
        for t_sub in itertools.combinations(best_block, t):
            idx = t_sub_to_idx[t_sub]
            if covered[idx] == 0:
                covered[idx] = 1
                uncovered_count -= 1

    # Reduction Phase: iteratively remove redundant blocks
    if blocks and time_ok():
        # Precompute t-subset index lists for each block
        block_t_subs = []
        for b in blocks:
            subs = [t_sub_to_idx[ts] for ts in itertools.combinations(b, t)]
            block_t_subs.append(subs)

        # Compute cover count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for subs in block_t_subs:
            for idx in subs:
                cover_count[idx] += 1

        changed = True
        while changed and time_ok():
            changed = False
            new_blocks = []
            new_block_t_subs = []
            new_block_masks = []
            for i, subs in enumerate(block_t_subs):
                keep = False
                for idx in subs:
                    if cover_count[idx] == 1:
                        keep = True
                        break
                if keep:
                    new_blocks.append(blocks[i])
                    new_block_t_subs.append(subs)
                    new_block_masks.append(block_masks[i])
                else:
                    changed = True
                    for idx in subs:
                        cover_count[idx] -= 1

            blocks = new_blocks
            block_t_subs = new_block_t_subs
            block_masks = new_block_masks

        # Local search: try to replace one block with two that cover more
        # (simple 1-exchange heuristic for further reduction)
        if time_ok() and len(blocks) > 1:
            # Recompute cover counts
            cover_count = np.zeros(n_t, dtype=np.int32)
            for subs in block_t_subs:
                for idx in subs:
                    cover_count[idx] += 1

            improved = True
            while improved and time_ok():
                improved = False
                # For each block, see if removing it and adding a better
                # replacement reduces total blocks (only try if we can
                # find a single block covering all its unique t-subsets)
                for i in range(len(blocks)):
                    if not time_ok():
                        break
                    subs = block_t_subs[i]
                    # Find t-subsets uniquely covered by block i
                    unique = [idx for idx in subs if cover_count[idx] == 1]
                    if not unique:
                        # Block is redundant; remove it
                        for idx in subs:
                            cover_count[idx] -= 1
                        blocks.pop(i)
                        block_t_subs.pop(i)
                        block_masks.pop(i)
                        improved = True
                        break
                    # Try to find a single block covering all unique t-subsets
                    # This is expensive; skip for now (handled by redundancy above)

    return blocks
# EVOLVE-BLOCK-END
