"""Covering design C(19,8,5) with greedy construction, redundancy removal, and swap optimization."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np
from collections import Counter


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute all t-subsets and map them to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Helper: time check
    def time_ok():
        return time.time() < deadline

    # --- Phase 1: Greedy construction with NumPy-vectorized gain ---
    # We track coverage as a numpy array of counts (0 = uncovered)
    coverage = np.zeros(n_t, dtype=np.int32)
    uncovered_mask = np.ones(n_t, dtype=bool)
    uncovered_count = n_t
    blocks = []

    # Precompute for each point, the list of t-subsets it participates in
    # point_to_tsubs[p] = list of indices of t-subsets containing p
    point_to_tsubs = [[] for _ in range(v)]
    for i, sub in enumerate(all_t_subs):
        for p in sub:
            point_to_tsubs[p].append(i)

    # Precompute for each pair (or t-1 subset), the t-subsets that extend it
    # For gain computation: given current block B, gain of adding point x
    # = number of uncovered t-subsets that contain x and t-1 elements from B

    while uncovered_count > 0 and time_ok():
        # Start a new block with the point in the most-uncovered t-subset
        # Pick the first uncovered t-subset (arbitrary) to seed
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            break
        seed_sub = all_t_subs[uncovered_indices[0]]
        block = list(seed_sub)

        # Greedily add points to maximize coverage of uncovered t-subsets
        while len(block) < k and time_ok():
            best_x = -1
            best_gain = 0
            for x in range(v):
                if x in block:
                    continue
                # Gain = number of uncovered t-subsets containing x and t-1 points from block
                # A t-subset containing x and t-1 from block: pick t-1 from block
                gain = 0
                for combo in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(combo + (x,)))
                    if sub in t_sub_to_idx:
                        idx = t_sub_to_idx[sub]
                        if uncovered_mask[idx]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1 or best_gain == 0:
                # No improvement possible; add any remaining point
                for x in range(v):
                    if x not in block:
                        block.append(x)
                        break
            else:
                block.append(best_x)

        block.sort()
        blocks.append(block)

        # Update coverage
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            if uncovered_mask[idx]:
                uncovered_count -= 1
                uncovered_mask[idx] = False
            coverage[idx] += 1

    # --- Phase 2: Redundancy removal (remove blocks that don't uniquely cover anything) ---
    def try_remove_block(block_idx):
        nonlocal coverage, uncovered_count
        block = blocks[block_idx]
        # Check if any t-subset is uniquely covered by this block
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            if coverage[idx] == 1:
                return False  # Can't remove
        # Safe to remove
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            coverage[idx] -= 1
            if coverage[idx] == 0:
                uncovered_count += 1
                uncovered_mask[idx] = True
        blocks.pop(block_idx)
        return True

    # Multiple passes of redundancy removal
    for _ in range(10):
        if not time_ok():
            break
        changed = True
        while changed and time_ok():
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if try_remove_block(i):
                    changed = True

    # --- Phase 3: Swap-based local optimization ---
    # Try to swap one point in a block for another to cover more uncovered subsets
    for _ in range(5):
        if not time_ok():
            break
        improved = True
        while improved and time_ok():
            improved = False
            for bi, block in enumerate(blocks):
                if not time_ok():
                    break
                for pos in range(k):
                    old_p = block[pos]
                    best_new_p = -1
                    best_net_gain = 0
                    for new_p in range(v):
                        if new_p in block:
                            continue
                        # Compute net gain: new uncovered covered - old uncovered lost
                        new_block = list(block)
                        new_block[pos] = new_p
                        new_block.sort()
                        net = 0
                        # t-subsets in new block not in old block
                        old_subs = set(itertools.combinations(block, t))
                        new_subs = set(itertools.combinations(new_block, t))
                        for sub in new_subs - old_subs:
                            idx = t_sub_to_idx[sub]
                            if uncovered_mask[idx]:
                                net += 1
                        for sub in old_subs - new_subs:
                            idx = t_sub_to_idx[sub]
                            if coverage[idx] == 1:
                                net -= 1
                        if net > best_net_gain:
                            best_net_gain = net
                            best_new_p = new_p
                    if best_new_p != -1 and best_net_gain > 0:
                        # Apply swap
                        old_subs = set(itertools.combinations(block, t))
                        block[pos] = best_new_p
                        block.sort()
                        blocks[bi] = block
                        new_subs = set(itertools.combinations(block, t))
                        for sub in new_subs - old_subs:
                            idx = t_sub_to_idx[sub]
                            if uncovered_mask[idx]:
                                uncovered_count -= 1
                                uncovered_mask[idx] = False
                            coverage[idx] += 1
                        for sub in old_subs - new_subs:
                            idx = t_sub_to_idx[sub]
                            coverage[idx] -= 1
                            if coverage[idx] == 0:
                                uncovered_count += 1
                                uncovered_mask[idx] = True
                        improved = True

    # --- Phase 4: Final redundancy removal after swaps ---
    for _ in range(5):
        if not time_ok():
            break
        changed = True
        while changed and time_ok():
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if try_remove_block(i):
                    changed = True

    # Convert to list of lists
    result = [list(b) for b in blocks]
    return result
# EVOLVE-BLOCK-END
