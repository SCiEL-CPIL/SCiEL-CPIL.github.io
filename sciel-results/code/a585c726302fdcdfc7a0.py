import itertools
import time
import random
import math
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    t_subset_to_idx = {sub: i for i, sub in enumerate(all_tsubsets)}

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    t_subsets_per_block = int(math.comb(k, t))

    # Precompute block_indices: shape (num_blocks, C(k, t))
    block_indices = np.empty((num_blocks, t_subsets_per_block), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        block_indices[i] = [t_subset_to_idx[sub] for sub in itertools.combinations(block, t)]

    # Precompute frequency of each t-subset across all blocks
    freq = np.bincount(block_indices.flatten(), minlength=num_tsubsets)
    inv_freq = 1.0 / (freq + 1e-9)

    best_blocks = None
    best_len = float('inf')

    # Run multiple strategies/restarts
    seed = 0
    while time.time() < deadline - 25:
        strategy = seed % 4

        if strategy == 0:
            blocks = _greedy_cover_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='greedy')
        elif strategy == 1:
            blocks = _greedy_cover_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='weighted', inv_freq=inv_freq)
        elif strategy == 2:
            blocks = _greedy_cover_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='randomized', inv_freq=inv_freq, seed=seed)
        else:
            blocks = _greedy_cover_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='ratio', inv_freq=inv_freq)

        if blocks is None:
            seed += 1
            continue

        # Post-processing: Remove redundant blocks
        blocks = _remove_redundant(blocks, t)

        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks

        seed += 1

        if best_len <= 76:
            break

    # Phase 2: Local Search on the best solution found
    if best_blocks is not None and time.time() < deadline - 5:
        best_blocks = _local_search(best_blocks, all_blocks, block_indices, num_tsubsets, t, deadline)

    if best_blocks is None:
        best_blocks = _greedy_cover_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='greedy')
        if best_blocks is None:
            best_blocks = []

    return best_blocks

def _greedy_cover_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='greedy', inv_freq=None, seed=0):
    uncovered = np.ones(num_tsubsets, dtype=bool)
    selected_indices = []

    while np.any(uncovered):
        if time.time() > deadline:
            return None

        if mode == 'greedy':
            gains = uncovered[block_indices].sum(axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_indices.append(best_idx)
            uncovered[block_indices[best_idx]] = False

        elif mode == 'weighted':
            w = np.where(uncovered, inv_freq, 0.0)
            gains = w[block_indices].sum(axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_indices.append(best_idx)
            uncovered[block_indices[best_idx]] = False

        elif mode == 'randomized':
            gains = uncovered[block_indices].sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 10:
                rng = random.Random(seed + len(selected_indices))
                candidates = rng.sample(list(candidates), 10)
            best_idx = candidates[np.random.choice(len(candidates))]
            selected_indices.append(best_idx)
            uncovered[block_indices[best_idx]] = False

        elif mode == 'ratio':
            # Ratio: newly covered triples / block size (k is constant, so this is same as greedy)
            # But let's use a different ratio: newly covered / (1 + overlap_count)
            # overlap_count = number of triples already covered
            gains = uncovered[block_indices].sum(axis=1)
            overlaps = (~uncovered)[block_indices].sum(axis=1)
            # Penalize overlap
            scores = gains / (1.0 + overlaps * 0.1)
            best_idx = np.argmax(scores)
            if gains[best_idx] == 0:
                break
            selected_indices.append(best_idx)
            uncovered[block_indices[best_idx]] = False

        else:
            raise ValueError("Unknown mode")

    if np.any(uncovered):
        return None

    return [list(all_blocks[i]) for i in selected_indices]

def _remove_redundant(blocks, t):
    if not blocks:
        return blocks

    # Count coverage of each t-subset
    cover_count = {}
    for block in blocks:
        for sub in itertools.combinations(block, t):
            cover_count[sub] = cover_count.get(sub, 0) + 1

    improved = list(blocks)
    n_blocks = len(improved)

    for i in range(n_blocks - 1, -1, -1):
        block = improved[i]
        redundant = True
        for sub in itertools.combinations(block, t):
            if cover_count.get(sub, 0) <= 1:
                redundant = False
                break

        if redundant:
            for sub in itertools.combinations(block, t):
                cover_count[sub] -= 1
            improved.pop(i)

    return improved

def _local_search(blocks, all_blocks, block_indices, num_tsubsets, t, deadline):
    if not blocks:
        return blocks

    current_blocks = list(blocks)
    current_len = len(current_blocks)

    for i in range(len(current_blocks) - 1, -1, -1):
        if time.time() > deadline:
            break

        remaining = current_blocks[:i] + current_blocks[i+1:]

        # Find uncovered triples
        covered = set()
        for b in remaining:
            for sub in itertools.combinations(b, t):
                covered.add(sub)

        all_subs = set(itertools.combinations(range(24), t))
        uncovered = all_subs - covered

        if not uncovered:
            current_blocks = remaining
            current_len -= 1
            continue

        # Greedy fill
        new_blocks = list(remaining)
        unc_set = set(uncovered)

        while unc_set:
            best_block = None
            best_gain = 0
            for b in all_blocks:
                gain = sum(1 for sub in itertools.combinations(b, t) if sub in unc_set)
                if gain > best_gain:
                    best_gain = gain
                    best_block = b

            if best_block is None or best_gain == 0:
                break

            new_blocks.append(list(best_block))
            for sub in itertools.combinations(best_block, t):
                unc_set.discard(sub)

        if len(new_blocks) < current_len:
            current_blocks = new_blocks
            current_len = len(new_blocks)

    return current_blocks
