"""
Covering design C(19,7,4) - bitset-accelerated greedy with local search.
"""
import itertools
import time
import random
import numpy as np
from collections import defaultdict


def _build_sub_to_idx(v, t):
    all_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {sub: i for i, sub in enumerate(all_subs)}
    return all_subs, sub_to_idx, len(all_subs)


def _build_block_masks(v, k, t, sub_to_idx, n_subs):
    """Build a numpy boolean matrix: rows = blocks (all C(v,k)), cols = t-subsets."""
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    # mask[i, j] = 1 if block i covers t-subset j
    mask = np.zeros((n_blocks, n_subs), dtype=np.uint8)
    for i, block in enumerate(all_blocks):
        for sub in itertools.combinations(block, t):
            mask[i, sub_to_idx[sub]] = 1
    return all_blocks, mask


def _greedy_construct(v, k, t, deadline, all_blocks, mask, sub_to_idx, n_subs):
    """
    Fast greedy covering using vectorized gain computation over all candidate blocks.
    """
    covered = np.zeros(n_subs, dtype=bool)
    selected = []
    selected_set = set()

    # Precompute row sums for speed
    row_sums = mask.sum(axis=1)  # number of t-subsets each block covers

    while not covered.all():
        if time.time() > deadline:
            break

        # Compute gain for each candidate block: number of uncovered subsets it covers
        uncovered_mask = ~covered  # shape (n_subs,)
        # gain[i] = sum of mask[i, :] & uncovered_mask
        gains = mask @ uncovered_mask.astype(np.uint8)

        # Exclude already selected blocks
        for idx in selected:
            gains[idx] = -1

        best_block_idx = int(np.argmax(gains))
        if gains[best_block_idx] <= 0:
            break

        selected.append(best_block_idx)
        selected_set.add(best_block_idx)
        covered |= mask[best_block_idx].astype(bool)

    return [list(all_blocks[i]) for i in selected]


def _greedy_construct_local(v, k, t, deadline, sub_to_idx, n_subs):
    """
    Local greedy: build each block by choosing points one at a time to maximize gain.
    Faster per-block but may find different (sometimes better) blocks.
    """
    covered = np.zeros(n_subs, dtype=bool)
    blocks = []

    # Precompute: for each (t-1)-subset, which t-subsets it extends to
    # tminus1_map: dict from frozenset of (t-1) elements -> list of (added_point, sub_idx)
    tminus1_map = defaultdict(list)
    all_subs_list = list(itertools.combinations(range(v), t))
    for idx, sub in enumerate(all_subs_list):
        for i in range(t):
            rest = tuple(sub[:i] + sub[i+1:])
            rest_key = tuple(sorted(rest))
            x = sub[i]
            tminus1_map[rest_key].append((x, idx))

    while not covered.all():
        if time.time() > deadline:
            break

        uncovered_indices = np.where(~covered)[0]
        if len(uncovered_indices) == 0:
            break
        start_sub = all_subs_list[uncovered_indices[0]]

        block = list(start_sub)
        block_set = set(block)

        while len(block) < k:
            if time.time() > deadline:
                break

            best_point = -1
            best_gain = -1

            for x in range(v):
                if x in block_set:
                    continue

                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    rest_key = tuple(sorted(rest))
                    if rest_key in tminus1_map:
                        for (pt, idx) in tminus1_map[rest_key]:
                            if pt == x and not covered[idx]:
                                gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_point = x

            if best_point == -1:
                break
            block.append(best_point)
            block_set.add(best_point)

        block.sort()
        blocks.append(block)

        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            covered[idx] = True

    return blocks


def _remove_redundant_blocks(blocks, v, t, deadline, sub_to_idx, n_subs):
    """Remove blocks that don't uniquely cover any t-subset."""
    if len(blocks) <= 1:
        return blocks

    changed = True
    while changed:
        changed = False
        coverage = np.zeros(n_subs, dtype=np.int32)
        for block in blocks:
            for sub in itertools.combinations(block, t):
                idx = sub_to_idx[sub]
                coverage[idx] += 1

        new_blocks = []
        for block in blocks:
            has_unique = False
            for sub in itertools.combinations(block, t):
                idx = sub_to_idx[sub]
                if coverage[idx] == 1:
                    has_unique = True
                    break

            if not has_unique:
                for sub in itertools.combinations(block, t):
                    idx = sub_to_idx[sub]
                    coverage[idx] -= 1
                changed = True
            else:
                new_blocks.append(block)

        if time.time() > deadline:
            return new_blocks

        blocks = new_blocks

    return blocks


def _local_search(blocks, v, k, t, deadline, sub_to_idx, n_subs, seed=42):
    """
    Local search: try removing blocks, swapping blocks, and perturbing blocks.
    """
    rng = np.random.RandomState(seed)
    current_blocks = [list(b) for b in blocks]
    best_blocks = [list(b) for b in current_blocks]
    best_size = len(current_blocks)

    coverage = np.zeros(n_subs, dtype=np.int32)
    for block in current_blocks:
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            coverage[idx] += 1

    def block_subs(block):
        return list(itertools.combinations(block, t))

    temp = 1.0
    cooling = 0.999
    min_temp = 0.001

    while time.time() < deadline and temp > min_temp:
        move_type = rng.randint(0, 4)

        if move_type <= 1 and len(current_blocks) > 1:
            # Try to remove a block
            idx = rng.randint(0, len(current_blocks))
            block = current_blocks[idx]
            subs = block_subs(block)

            can_remove = True
            for sub in subs:
                sidx = sub_to_idx[sub]
                if coverage[sidx] <= 1:
                    can_remove = False
                    break

            if can_remove:
                for sub in subs:
                    sidx = sub_to_idx[sub]
                    coverage[sidx] -= 1
                current_blocks.pop(idx)

                if len(current_blocks) < best_size:
                    best_blocks = [list(b) for b in current_blocks]
                    best_size = len(current_blocks)

            temp *= cooling

        elif move_type == 2 and len(current_blocks) > 1:
            # Swap: replace a block
            idx = rng.randint(0, len(current_blocks))
            old_block = current_blocks[idx]
            old_subs = block_subs(old_block)

            can_remove = True
            for sub in old_subs:
                sidx = sub_to_idx[sub]
                if coverage[sidx] <= 1:
                    can_remove = False
                    break

            if not can_remove:
                temp *= cooling
                continue

            for sub in old_subs:
                sidx = sub_to_idx[sub]
                coverage[sidx] -= 1

            best_block = None
            best_gain = -1
            for _ in range(12):
                candidate = sorted(rng.choice(v, size=k, replace=False).tolist())
                gain = 0
                for sub in itertools.combinations(candidate, t):
                    sidx = sub_to_idx[sub]
                    if coverage[sidx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = candidate

            if best_block is not None and best_gain > 0:
                for sub in itertools.combinations(best_block, t):
                    sidx = sub_to_idx[sub]
                    coverage[sidx] += 1
                current_blocks[idx] = best_block
            else:
                for sub in old_subs:
                    sidx = sub_to_idx[sub]
                    coverage[sidx] += 1

            temp *= cooling

        else:
            # Perturb: modify a random block
            if len(current_blocks) == 0:
                temp *= cooling
                continue

            idx = rng.randint(0, len(current_blocks))
            old_block = current_blocks[idx]
            old_subs = block_subs(old_block)

            new_block = list(old_block)
            num_changes = rng.randint(1, 2)
            for _ in range(num_changes):
                pos = rng.randint(0, k)
                new_val = rng.randint(0, v)
                if new_val not in new_block:
                    new_block[pos] = new_val

            new_block.sort()

            if new_block == old_block:
                temp *= cooling
                continue

            new_subs = block_subs(new_block)
            new_sub_set = set(new_subs)

            valid = True
            for sub in old_subs:
                sidx = sub_to_idx[sub]
                if coverage[sidx] <= 1 and sub not in new_sub_set:
                    valid = False
                    break

            if valid:
                for sub in old_subs:
                    sidx = sub_to_idx[sub]
                    coverage[sidx] -= 1
                for sub in new_subs:
                    sidx = sub_to_idx[sub]
                    coverage[sidx] += 1
                current_blocks[idx] = new_block

            temp *= cooling

    return best_blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112  # Leave 8s margin

    all_subs, sub_to_idx, n_subs = _build_sub_to_idx(v, t)

    # Strategy 1: Global greedy over all C(v,k) blocks using bitset matrix
    all_blocks, mask = _build_block_masks(v, k, t, sub_to_idx, n_subs)
    blocks1 = _greedy_construct(v, k, t, deadline, all_blocks, mask, sub_to_idx, n_subs)
    blocks1 = _remove_redundant_blocks(blocks1, v, t, deadline, sub_to_idx, n_subs)

    # Strategy 2: Local greedy (build blocks point by point)
    remaining = deadline - time.time()
    blocks2 = []
    if remaining > 10:
        dl2 = start_time + min(remaining - 5, 40)
        blocks2 = _greedy_construct_local(v, k, t, dl2, sub_to_idx, n_subs)
        blocks2 = _remove_redundant_blocks(blocks2, v, t, deadline, sub_to_idx, n_subs)

    # Pick the better starting point
    if len(blocks2) < len(blocks1):
        blocks = blocks2
    else:
        blocks = blocks1

    # Local search with multiple seeds
    remaining = deadline - time.time()
    if remaining > 15:
        dl_ls = start_time + remaining - 5
        blocks = _local_search(blocks, v, k, t, dl_ls, sub_to_idx, n_subs, seed=42)

    # Final cleanup
    remaining = deadline - time.time()
    if remaining > 5:
        blocks = _remove_redundant_blocks(blocks, v, t, deadline, sub_to_idx, n_subs)

    # Verify and fix coverage
    coverage = np.zeros(n_subs, dtype=np.int32)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            coverage[idx] += 1

    while not (coverage > 0).all() and time.time() < deadline:
        uncovered_idx = int(np.where(coverage == 0)[0][0])
        sub = all_subs[uncovered_idx]
        block = list(sub)
        block_set = set(block)

        while len(block) < k:
            best_point = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    candidate = tuple(sorted(rest + (x,)))
                    if candidate in sub_to_idx:
                        sidx = sub_to_idx[candidate]
                        if coverage[sidx] == 0:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point == -1:
                break
            block.append(best_point)
            block_set.add(best_point)

        block.sort()
        blocks.append(block)
        for sub2 in itertools.combinations(block, t):
            sidx = sub_to_idx[sub2]
            coverage[sidx] += 1

    return blocks
