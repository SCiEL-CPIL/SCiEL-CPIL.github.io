import itertools
import random
import time
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    # Step 1: Generate all t-subsets and index them
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}

    # Step 2: Generate all k-blocks and compute which t-subsets each covers
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)

    # For each k-block, compute the set of t-subset indices it covers
    block_t_subsets = []
    for block in k_blocks:
        covered = set()
        for combo in itertools.combinations(block, t):
            covered.add(t_index[combo])
        block_t_subsets.append(covered)

    if elapsed() > deadline - 60:
        # Fallback: simple unweighted greedy
        uncovered = set(range(num_t))
        selected = []
        while uncovered and elapsed() < deadline - 10:
            best_idx = -1
            best_gain = -1
            for i in range(num_k):
                if i in selected:
                    continue
                gain = len(block_t_subsets[i] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1 or best_gain == 0:
                break
            selected.append(best_idx)
            uncovered -= block_t_subsets[best_idx]
        result = [list(k_blocks[i]) for i in selected]
        # Remove redundant blocks
        if elapsed() < deadline - 5:
            cover_count = np.zeros(num_t, dtype=np.int32)
            for i in selected:
                for idx in block_t_subsets[i]:
                    cover_count[idx] += 1
            new_selected = []
            for i in selected:
                needed = False
                for idx in block_t_subsets[i]:
                    if cover_count[idx] == 1:
                        needed = True
                        break
                if needed:
                    new_selected.append(i)
                    for idx in block_t_subsets[i]:
                        cover_count[idx] -= 1
                # else: redundant, already decremented? No, we didn't decrement yet
            # Redo properly
            cover_count = np.zeros(num_t, dtype=np.int32)
            for i in selected:
                for idx in block_t_subsets[i]:
                    cover_count[idx] += 1
            new_selected = []
            for i in selected:
                needed = False
                for idx in block_t_subsets[i]:
                    if cover_count[idx] == 1:
                        needed = True
                        break
                if needed:
                    new_selected.append(i)
            selected = new_selected
            result = [list(k_blocks[i]) for i in selected]
        return result

    # Step 3: Build reverse mapping: for each t-subset, which blocks cover it
    t_to_blocks = [[] for _ in range(num_t)]
    for i, covered in enumerate(block_t_subsets):
        for idx in covered:
            t_to_blocks[idx].append(i)

    # Compute initial coverage count per t-subset
    t_cover_count = np.zeros(num_t, dtype=np.int32)
    for i in range(num_k):
        for idx in block_t_subsets[i]:
            t_cover_count[idx] += 1

    # Step 4: Weighted greedy set cover
    # Weight of t-subset j = 1 / (number of blocks that cover it)
    # This prioritizes "rare" t-subsets
    safe_counts = np.maximum(t_cover_count, 1).astype(np.float64)
    t_weights = 1.0 / safe_counts

    # Initialize gain for each block: sum of weights of t-subsets it covers
    gain = np.zeros(num_k, dtype=np.float64)
    for i in range(num_k):
        if block_t_subsets[i]:
            gain[i] = np.sum(t_weights[list(block_t_subsets[i])])

    # Track uncovered
    uncovered = np.ones(num_t, dtype=np.bool_)
    uncovered_count = num_t
    selected = []
    selected_mask = np.zeros(num_k, dtype=np.bool_)

    while uncovered_count > 0 and elapsed() < deadline - 30:
        # Find best block
        best_idx = -1
        best_gain = -1.0
        for i in range(num_k):
            if selected_mask[i]:
                continue
            if gain[i] > best_gain:
                best_gain = gain[i]
                best_idx = i

        if best_idx == -1 or best_gain <= 0:
            break

        selected.append(best_idx)
        selected_mask[best_idx] = True

        # Update uncovered and gains
        for idx in block_t_subsets[best_idx]:
            if uncovered[idx]:
                uncovered[idx] = False
                uncovered_count -= 1
                w = t_weights[idx]
                # Subtract weight from all blocks covering this t-subset
                for b in t_to_blocks[idx]:
                    if not selected_mask[b]:
                        gain[b] -= w

    if elapsed() > deadline - 15:
        # Not enough time for post-processing, return what we have
        result = [list(k_blocks[i]) for i in selected]
        return result

    # Step 5: Post-processing - remove redundant blocks
    # Build coverage count
    cover_count = np.zeros(num_t, dtype=np.int32)
    for i in selected:
        for idx in block_t_subsets[i]:
            cover_count[idx] += 1

    # Iteratively remove redundant blocks
    changed = True
    while changed and elapsed() < deadline - 5:
        changed = False
        new_selected = []
        for i in selected:
            needed = False
            for idx in block_t_subsets[i]:
                if cover_count[idx] == 1:
                    needed = True
                    break
            if needed:
                new_selected.append(i)
            else:
                changed = True
                for idx in block_t_subsets[i]:
                    cover_count[idx] -= 1
        if changed:
            selected = new_selected

    result = [list(k_blocks[i]) for i in selected]

    # Step 6: Try to add back blocks if coverage is incomplete (shouldn't happen but safety)
    if elapsed() < deadline - 2:
        # Check if all t-subsets are covered
        cover_count_final = np.zeros(num_t, dtype=np.int32)
        for i in selected:
            for idx in block_t_subsets[i]:
                cover_count_final[idx] += 1
        if not np.all(cover_count_final > 0):
            # Add blocks to cover missing t-subsets
            uncovered_final = set(np.where(cover_count_final == 0)[0])
            while uncovered_final and elapsed() < deadline - 1:
                best_idx = -1
                best_gain = -1
                for i in range(num_k):
                    if i in selected:
                        continue
                    gain_val = len(block_t_subsets[i] & uncovered_final)
                    if gain_val > best_gain:
                        best_gain = gain_val
                        best_idx = i
                if best_idx == -1 or best_gain == 0:
                    break
                selected.append(best_idx)
                for idx in block_t_subsets[best_idx]:
                    if idx in uncovered_final:
                        uncovered_final.discard(idx)
                result = [list(k_blocks[i]) for i in selected]

    return result
