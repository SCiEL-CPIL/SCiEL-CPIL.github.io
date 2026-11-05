import itertools
import time
import random
from collections import defaultdict
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 120s

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    t_subset_to_idx = {sub: i for i, sub in enumerate(all_tsubsets)}

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # Precompute block_indices: for each block, the indices of its t-subsets
    t_subsets_per_block = int(math.comb(k, t))
    block_indices = np.empty((num_blocks, t_subsets_per_block), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        block_indices[i] = [t_subset_to_idx[sub] for sub in itertools.combinations(block, t)]

    # Precompute frequency of each t-subset across all blocks
    freq = np.bincount(block_indices.flatten(), minlength=n_tsubsets)
    inv_freq = 1.0 / (freq + 1e-9)

    best_blocks = None
    best_len = float('inf')

    # Phase 1: Multiple greedy strategies
    trial = 0
    while time.time() < deadline - 25:
        trial += 1
        seed = trial * 7919 + 42

        # Strategy 1: Vectorized greedy (max gain)
        blocks = _greedy_vectorized(all_blocks, block_indices, n_tsubsets, deadline, mode='greedy')
        if blocks is not None:
            blocks = _remove_redundant(blocks, block_indices, n_tsubsets, deadline)
            blocks = _local_search(blocks, v, k, t, all_tsubsets, block_indices, n_tsubsets, deadline, seed)
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
                if best_len <= 76:
                    break

        # Strategy 2: Weighted greedy (inverse frequency)
        if time.time() < deadline - 25:
            blocks2 = _greedy_vectorized(all_blocks, block_indices, n_tsubsets, deadline, mode='weighted', inv_freq=inv_freq)
            if blocks2 is not None:
                blocks2 = _remove_redundant(blocks2, block_indices, n_tsubsets, deadline)
                blocks2 = _local_search(blocks2, v, k, t, all_tsubsets, block_indices, n_tsubsets, deadline, seed + 1)
                if len(blocks2) < best_len:
                    best_len = len(blocks2)
                    best_blocks = blocks2
                    if best_len <= 76:
                        break

        # Strategy 3: Randomized greedy
        if time.time() < deadline - 25:
            blocks3 = _greedy_vectorized(all_blocks, block_indices, n_tsubsets, deadline, mode='randomized', inv_freq=inv_freq, seed=seed)
            if blocks3 is not None:
                blocks3 = _remove_redundant(blocks3, block_indices, n_tsubsets, deadline)
                blocks3 = _local_search(blocks3, v, k, t, all_tsubsets, block_indices, n_tsubsets, deadline, seed + 2)
                if len(blocks3) < best_len:
                    best_len = len(blocks3)
                    best_blocks = blocks3
                    if best_len <= 76:
                        break

        # Strategy 4: Flow-based refinement on best
        if trial >= 4 and best_blocks is not None and time.time() < deadline - 15:
            polished = _flow_based_refinement(best_blocks, all_blocks, block_indices, n_tsubsets, deadline)
            if len(polished) < best_len:
                best_len = len(polished)
                best_blocks = polished

    # Phase 2: Deep local search on the best
    if best_blocks is not None and time.time() < deadline - 5:
        polished = _deep_local_search(best_blocks, v, k, t, all_tsubsets, block_indices, n_tsubsets, deadline)
        if len(polished) < best_len:
            best_len = len(polished)
            best_blocks = polished

    if best_blocks is None:
        # Fallback: deterministic greedy
        blocks = _greedy_vectorized(all_blocks, block_indices, n_tsubsets, deadline, mode='greedy')
        if blocks is not None:
            best_blocks = _remove_redundant(blocks, block_indices, n_tsubsets, deadline)
        else:
            best_blocks = []

    return [list(b) for b in best_blocks]


def _greedy_vectorized(all_blocks, block_indices, n_tsubsets, deadline, mode='greedy', inv_freq=None, seed=0):
    """
    Vectorized greedy set cover.
    """
    uncovered = np.ones(n_tsubsets, dtype=bool)
    selected_indices = []
    rng = random.Random(seed)

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
                candidates = rng.sample(list(candidates), 10)
            best_idx = candidates[rng.randint(0, len(candidates) - 1)]
            selected_indices.append(best_idx)
            uncovered[block_indices[best_idx]] = False

    if np.any(uncovered):
        return None

    return [list(all_blocks[i]) for i in selected_indices]


def _remove_redundant(blocks, block_indices, n_tsubsets, deadline):
    """
    Remove blocks that do not cover any unique t-subset.
    """
    if not blocks:
        return blocks

    # Map selected blocks to their t-subset indices
    block_to_tsubs = []
    for block in blocks:
        tsubs = [t_subset_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_to_tsubs.append(tsubs)

    cover_count = np.zeros(n_tsubsets, dtype=np.int32)
    for tsubs in block_to_tsubs:
        cover_count[tsubs] += 1

    improved = list(blocks)
    improved_tsubs = list(block_to_tsubs)

    for i in range(len(improved) - 1, -1, -1):
        if time.time() > deadline:
            break

        tsubs = improved_tsubs[i]
        if np.all(cover_count[tsubs] > 1):
            cover_count[tsubs] -= 1
            improved.pop(i)
            improved_tsubs.pop(i)

    return improved


def _local_search(blocks, v, k, t, all_tsubsets, block_indices, n_tsubsets, deadline, seed_val):
    """
    Local search: try to replace blocks to reduce total count.
    """
    if not blocks:
        return blocks

    rng = random.Random(seed_val)
    improved = list(blocks)

    # Build cover count
    cover_count = defaultdict(int)
    for block in improved:
        for s in itertools.combinations(block, t):
            cover_count[s] += 1

    n_blocks = len(improved)
    max_iterations = 150

    for iteration in range(max_iterations):
        if time.time() > deadline:
            break

        any_change = False

        for i in range(n_blocks):
            if time.time() > deadline:
                break

            old_block = improved[i]
            old_tsubs = list(itertools.combinations(old_block, t))
            old_tsub_set = set(old_tsubs)

            unique_subs = [s for s in old_tsubs if cover_count[s] == 1]

            if not unique_subs:
                for s in old_tsubs:
                    cover_count[s] -= 1
                    if cover_count[s] == 0:
                        del cover_count[s]
                improved.pop(i)
                n_blocks -= 1
                any_change = True
                break

            best_new_block = None
            best_score = -1

            for _ in range(25):
                if time.time() > deadline:
                    break

                new_block = list(old_block)
                remove_count = rng.randint(1, min(2, k))
                to_remove = rng.sample(new_block, remove_count)
                for x in to_remove:
                    new_block.remove(x)
                while len(new_block) < k:
                    x = rng.randint(0, v - 1)
                    if x not in new_block:
                        new_block.append(x)
                cand = tuple(sorted(new_block))

                if cand == old_block:
                    continue

                cand_tsubs = set(itertools.combinations(cand, t))

                if not all(s in cand_tsubs for s in unique_subs):
                    continue

                new_subs = cand_tsubs - old_tsub_set
                gain = len(new_subs)

                redundancy_bonus = 0
                for s in new_subs:
                    if cover_count.get(s, 0) >= 1:
                        redundancy_bonus += 1

                score = gain + redundancy_bonus * 0.5

                if score > best_score:
                    best_score = score
                    best_new_block = cand

            if best_new_block is not None and best_score > 0:
                for s in old_tsubs:
                    cover_count[s] -= 1
                    if cover_count[s] == 0:
                        del cover_count[s]
                for s in itertools.combinations(best_new_block, t):
                    cover_count[s] += 1
                improved[i] = best_new_block
                any_change = True
                break

        if not any_change:
            break

    improved = _remove_redundant(improved, block_indices, n_tsubsets, deadline)

    return improved


def _flow_based_refinement(blocks, all_blocks, block_indices, n_tsubsets, deadline):
    """
    Flow-based refinement: use min-cut analysis to identify redundant blocks.
    """
    if not blocks:
        return blocks

    improved = list(blocks)
    best = improved
    best_len = len(improved)

    # Build cover count
    cover_count = defaultdict(int)
    for block in improved:
        for s in itertools.combinations(block, t):
            cover_count[s] += 1

    # Try to remove blocks that are "redundant" in a flow sense
    # A block is redundant if all its t-subsets are covered by other blocks
    for _ in range(5):
        if time.time() > deadline:
            break

        changed = False
        for i in range(len(improved) - 1, -1, -1):
            if time.time() > deadline:
                break

            block = improved[i]
            tsubs = list(itertools.combinations(block, t))
            redundant = True
            for s in tsubs:
                if cover_count[s] <= 1:
                    redundant = False
                    break

            if redundant:
                for s in tsubs:
                    cover_count[s] -= 1
                improved.pop(i)
                changed = True

        if not changed:
            break

    if len(improved) < best_len:
        best = improved
        best_len = len(improved)

    return best


def _deep_local_search(blocks, v, k, t, all_tsubsets, block_indices, n_tsubsets, deadline):
    """
    More intensive local search for the best solution found.
    """
    if not blocks:
        return blocks

    improved = list(blocks)
    best = improved
    best_len = len(improved)

    for trial in range(8):
        if time.time() > deadline:
            break

        seed = trial * 104729 + 7
        candidate = _local_search(improved, v, k, t, all_tsubsets, block_indices, n_tsubsets, deadline, seed)
        candidate = _remove_redundant(candidate, block_indices, n_tsubsets, deadline)

        if len(candidate) < best_len:
            best = candidate
            best_len = len(candidate)
            improved = candidate

    return best


import math
