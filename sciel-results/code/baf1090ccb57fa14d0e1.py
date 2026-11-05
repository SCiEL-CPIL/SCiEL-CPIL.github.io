"""Combined covering design: greedy construction + vectorized redundancy removal + swap-based local search."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    uncovered = set(range(n_t))
    blocks = []

    def time_ok():
        return time.time() < deadline

    # Phase 1: Greedy construction
    while uncovered and time_ok():
        first_idx = min(uncovered)
        block = list(all_t_subs[first_idx])
        block_set = set(block)

        while len(block) < k and time_ok():
            best_x = -1
            best_gain = -1
            t_minus_1 = list(itertools.combinations(block, t - 1))

            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in t_minus_1:
                    cand = tuple(sorted(rest + (x,)))
                    idx = t_sub_to_idx.get(cand)
                    if idx is not None and idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            if best_x == -1:
                for x in range(v):
                    if len(block) >= k:
                        break
                    if x not in block_set:
                        block.append(x)
                        block_set.add(x)
                break

            block.append(best_x)
            block_set.add(best_x)

        block.sort()
        blocks.append(block)

        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx.get(t_sub)
            if idx is not None:
                uncovered.discard(idx)

    if uncovered:
        return blocks

    n_blocks = len(blocks)
    if n_blocks == 0:
        return blocks

    # Precompute: for each block, the list of t-subset indices it covers
    block_t_indices = []
    for b in blocks:
        idxs = []
        for t_sub in itertools.combinations(b, t):
            idx = t_sub_to_idx.get(t_sub)
            if idx is not None:
                idxs.append(idx)
        block_t_indices.append(np.array(idxs, dtype=np.int32))

    # Build reverse index: for each t-subset, which blocks cover it
    # This is needed for swap operations
    # We'll build it lazily or as a list of lists
    subset_to_blocks = [[] for _ in range(n_t)]
    for b_idx, idxs in enumerate(block_t_indices):
        for s_idx in idxs:
            subset_to_blocks[s_idx].append(b_idx)

    # Initialize coverage count array
    cov = np.zeros(n_t, dtype=np.int32)
    for idxs in block_t_indices:
        if len(idxs) > 0:
            np.add.at(cov, idxs, 1)

    # Phase 2: Vectorized redundancy removal
    removed = np.zeros(n_blocks, dtype=bool)
    changed = True
    while changed and time_ok():
        changed = False
        for i in range(n_blocks):
            if removed[i]:
                continue
            idxs = block_t_indices[i]
            if len(idxs) == 0:
                removed[i] = True
                changed = True
                continue
            counts = cov[idxs]
            if np.any(counts == 1):
                continue
            removed[i] = True
            changed = True
            np.add.at(cov, idxs, -1)

    # Build active block list after redundancy removal
    active_indices = [i for i in range(n_blocks) if not removed[i]]
    if not active_indices:
        return []

    # Rebuild data structures for active blocks only
    # active_blocks: list of block tuples
    # active_block_t_indices: list of arrays of t-subset indices
    # active_subset_to_blocks: for each t-subset, list of active block positions
    n_active = len(active_indices)
    active_blocks = [blocks[i] for i in active_indices]
    active_block_t_indices = [block_t_indices[i] for i in active_indices]

    # Rebuild coverage counts for active blocks
    active_cov = np.zeros(n_t, dtype=np.int32)
    for idxs in active_block_t_indices:
        if len(idxs) > 0:
            np.add.at(active_cov, idxs, 1)

    # Rebuild subset_to_blocks for active blocks
    active_subset_to_blocks = [[] for _ in range(n_t)]
    for pos, idxs in enumerate(active_block_t_indices):
        for s_idx in idxs:
            active_subset_to_blocks[s_idx].append(pos)

    # Phase 3: Swap-based local search
    # Strategy: repeatedly try to swap blocks to enable further deletions
    best_set = list(active_blocks)
    best_len = len(active_blocks)

    # Precompute all possible k-subsets as tuples for swap candidates
    # This is C(v, k) which for v=19, k=8 is 75582 - manageable
    all_k_blocks = list(itertools.combinations(range(v), k))
    # Map from tuple to its t-subset indices
    all_block_t_indices = {}
    for blk in all_k_blocks:
        idxs = []
        for t_sub in itertools.combinations(blk, t):
            idx = t_sub_to_idx.get(t_sub)
            if idx is not None:
                idxs.append(idx)
        all_block_t_indices[blk] = np.array(idxs, dtype=np.int32)

    # For fast lookup: which all_k_blocks contain a given t-subset
    # t_sub_to_all_blocks: for each t-subset index, list of block tuples that contain it
    t_sub_to_all_blocks = [[] for _ in range(n_t)]
    for blk, idxs in all_block_t_indices.items():
        for s_idx in idxs:
            t_sub_to_all_blocks[s_idx].append(blk)

    # Current state for local search
    cur_blocks = list(active_blocks)
    cur_block_t_indices = list(active_block_t_indices)
    cur_cov = active_cov.copy()
    cur_subset_to_blocks = [list(lst) for lst in active_subset_to_blocks]

    def try_deletion_pass(cur_blocks, cur_block_t_indices, cur_cov, cur_subset_to_blocks):
        """Try to remove redundant blocks. Returns (success, new_blocks, new_bti, new_cov, new_stb)."""
        changed = True
        while changed:
            changed = False
            # Try random order
            order = list(range(len(cur_blocks)))
            random.shuffle(order)
            for pos in order:
                idxs = cur_block_t_indices[pos]
                if len(idxs) == 0:
                    # Remove
                    np.add.at(cur_cov, idxs, -1)
                    for s_idx in idxs:
                        cur_subset_to_blocks[s_idx].remove(pos)
                    # Shift indices in subset_to_blocks for blocks after pos
                    for s_idx in range(n_t):
                        stb = cur_subset_to_blocks[s_idx]
                        for j in range(len(stb)):
                            if stb[j] > pos:
                                stb[j] -= 1
                    cur_blocks.pop(pos)
                    cur_block_t_indices.pop(pos)
                    changed = True
                    break
                counts = cur_cov[idxs]
                if np.any(counts == 1):
                    continue
                # Redundant
                np.add.at(cur_cov, idxs, -1)
                for s_idx in idxs:
                    cur_subset_to_blocks[s_idx].remove(pos)
                for s_idx in range(n_t):
                    stb = cur_subset_to_blocks[s_idx]
                    for j in range(len(stb)):
                        if stb[j] > pos:
                            stb[j] -= 1
                cur_blocks.pop(pos)
                cur_block_t_indices.pop(pos)
                changed = True
                break
        return cur_blocks, cur_block_t_indices, cur_cov, cur_subset_to_blocks

    def perform_swap(cur_blocks, cur_block_t_indices, cur_cov, cur_subset_to_blocks, pos, new_block_tuple):
        """Swap block at pos with new_block_tuple. Returns updated structures."""
        old_block = cur_blocks[pos]
        old_idxs = cur_block_t_indices[pos]
        new_idxs = all_block_t_indices[new_block_tuple]

        # Remove old block's coverage
        np.add.at(cur_cov, old_idxs, -1)
        for s_idx in old_idxs:
            if pos in cur_subset_to_blocks[s_idx]:
                cur_subset_to_blocks[s_idx].remove(pos)

        # Add new block's coverage
        np.add.at(cur_cov, new_idxs, 1)
        for s_idx in new_idxs:
            cur_subset_to_blocks[s_idx].append(pos)

        cur_blocks[pos] = list(new_block_tuple)
        cur_block_t_indices[pos] = new_idxs
        return cur_blocks, cur_block_t_indices, cur_cov, cur_subset_to_blocks

    # Local search loop
    while time_ok():
        # Deletion pass
        cur_blocks, cur_block_t_indices, cur_cov, cur_subset_to_blocks = try_deletion_pass(
            cur_blocks, cur_block_t_indices, cur_cov, cur_subset_to_blocks
        )

        if len(cur_blocks) < best_len:
            best_len = len(cur_blocks)
            best_set = list(cur_blocks)

        if not cur_blocks:
            break

        # Swap pass
        n_cur = len(cur_blocks)
        indices = list(range(n_cur))
        random.shuffle(indices)

        swaps_done = 0
        max_swaps = 30

        for pos in indices:
            if time.time() > deadline:
                break
            if swaps_done >= max_swaps:
                break

            old_block = cur_blocks[pos]
            old_idxs = cur_block_t_indices[pos]

            # Find unique subsets (covered only by this block)
            unique_subs = []
            for s_idx in old_idxs:
                if cur_cov[s_idx] == 1:
                    unique_subs.append(s_idx)

            if not unique_subs:
                # Not critical: try swapping with a random block
                new_block = random.choice(all_k_blocks)
                if new_block == tuple(old_block):
                    continue
                if new_block in cur_blocks:
                    continue
                # Perform swap
                cur_blocks, cur_block_t_indices, cur_cov, cur_subset_to_blocks = perform_swap(
                    cur_blocks, cur_block_t_indices, cur_cov, cur_subset_to_blocks, pos, new_block
                )
                swaps_done += 1
            else:
                # Critical: find candidates that cover all unique_subs
                candidates = None
                for s_idx in unique_subs:
                    if candidates is None:
                        candidates = set(t_sub_to_all_blocks[s_idx])
                    else:
                        candidates.intersection_update(t_sub_to_all_blocks[s_idx])
                        if not candidates:
                            break

                if candidates:
                    # Remove current block from candidates
                    candidates.discard(tuple(old_block))
                    if not candidates:
                        continue
                    # Pick a random candidate
                    new_block = random.choice(list(candidates))
                    if new_block in cur_blocks:
                        continue
                    cur_blocks, cur_block_t_indices, cur_cov, cur_subset_to_blocks = perform_swap(
                        cur_blocks, cur_block_t_indices, cur_cov, cur_subset_to_blocks, pos, new_block
                    )
                    swaps_done += 1

        # Check if we've hit the deadline
        if time.time() > deadline:
            break

    # Final deletion pass
    if time_ok():
        cur_blocks, cur_block_t_indices, cur_cov, cur_subset_to_blocks = try_deletion_pass(
            cur_blocks, cur_block_t_indices, cur_cov, cur_subset_to_blocks
        )
        if len(cur_blocks) < best_len:
            best_len = len(cur_blocks)
            best_set = list(cur_blocks)

    # Return the best set found
    return best_set
# EVOLVE-BLOCK-END
