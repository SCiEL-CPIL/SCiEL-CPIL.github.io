import itertools
import time
import random
from collections import Counter
import numpy as np
from itertools import combinations
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0

    # Precompute all t-subsets and map to indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {sub: i for i, sub in enumerate(all_tsubsets)}
    num_tsubsets = len(all_tsubsets)

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # Precompute block_indices: for each block, the indices of its t-subsets
    t_subsets_per_block = int(math.comb(k, t))
    block_indices = np.empty((num_blocks, t_subsets_per_block), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        block_indices[i] = [t_subset_to_idx[sub] for sub in itertools.combinations(block, t)]
        if time.time() > deadline - 10:
            break

    # Precompute frequency of each t-subset across all blocks
    freq = np.bincount(block_indices.flatten(), minlength=num_tsubsets)
    inv_freq = 1.0 / (freq + 1e-9)

    best_blocks = None
    best_len = float('inf')

    seed = 0
    # Run multiple strategies: greedy, weighted, UCB
    while time.time() < deadline - 25:
        strategy = seed % 3

        if strategy == 0:
            blocks = _greedy_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='greedy')
        elif strategy == 1:
            blocks = _greedy_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='weighted', inv_freq=inv_freq)
        else:
            blocks = _greedy_ucb(all_blocks, block_indices, num_tsubsets, deadline, inv_freq=inv_freq, seed=seed)

        if blocks is not None:
            blocks = _remove_redundant(blocks, v, k, t, deadline)
            blocks = _local_search(blocks, v, k, t, all_tsubsets, deadline, seed)
            blocks = _remove_redundant(blocks, v, k, t, deadline)

            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks

        seed += 1
        if best_len <= 76:
            break

    if best_blocks is None:
        return []

    return [list(b) for b in best_blocks]


def _greedy_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='greedy', inv_freq=None):
    """Vectorized greedy set cover."""
    uncovered = np.ones(num_tsubsets, dtype=bool)
    selected = []

    while np.any(uncovered):
        if time.time() > deadline:
            return None

        if mode == 'greedy':
            gains = uncovered[block_indices].sum(axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected.append(best_idx)
            uncovered[block_indices[best_idx]] = False

        elif mode == 'weighted':
            w = np.where(uncovered, inv_freq, 0.0)
            gains = w[block_indices].sum(axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected.append(best_idx)
            uncovered[block_indices[best_idx]] = False

    if np.any(uncovered):
        return None

    return [list(all_blocks[i]) for i in selected]


def _greedy_ucb(all_blocks, block_indices, num_tsubsets, deadline, inv_freq=None, seed=0):
    """
    UCB-based greedy: balance exploration and exploitation.
    UCB score = mean_gain + c * sqrt(ln(total_steps) / pull_count)
    """
    rng = random.Random(seed)
    uncovered = np.ones(num_tsubsets, dtype=bool)
    selected = []
    pull_counts = np.zeros(len(all_blocks), dtype=np.float64)
    total_reward = np.zeros(len(all_blocks), dtype=np.float64)
    c = 1.414  # sqrt(2)
    steps = 0

    while np.any(uncovered):
        if time.time() > deadline:
            return None

        steps += 1
        gains = uncovered[block_indices].sum(axis=1)

        # Compute UCB scores
        # For blocks never pulled, give them a high exploration bonus
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_gain = np.where(pull_counts > 0, total_reward / pull_counts, gains)
            exploration = np.where(pull_counts > 0,
                                   c * np.sqrt(np.log(steps + 1) / pull_counts),
                                   1e6)  # High bonus for unexplored
            ucb_scores = mean_gain + exploration

        # Only consider blocks with positive gain OR unexplored
        mask = (gains > 0) | (pull_counts == 0)
        ucb_scores[~mask] = -1e9

        best_idx = np.argmax(ucb_scores)
        if ucb_scores[best_idx] < 0:
            break

        selected.append(best_idx)
        pull_counts[best_idx] += 1
        total_reward[best_idx] += gains[best_idx]
        uncovered[block_indices[best_idx]] = False

    if np.any(uncovered):
        return None

    return [list(all_blocks[i]) for i in selected]


def _remove_redundant(blocks, v, k, t, deadline):
    """Remove blocks that don't cover any unique t-subset."""
    if not blocks:
        return blocks

    cover_count = Counter()
    for block in blocks:
        for sub in itertools.combinations(block, t):
            cover_count[sub] += 1

    improved = list(blocks)
    changed = True

    while changed:
        if time.time() > deadline:
            break
        changed = False
        for i in range(len(improved) - 1, -1, -1):
            if time.time() > deadline:
                break
            block = improved[i]
            is_redundant = True
            for sub in itertools.combinations(block, t):
                if cover_count[sub] <= 1:
                    is_redundant = False
                    break

            if is_redundant:
                for sub in itertools.combinations(block, t):
                    cover_count[sub] -= 1
                improved.pop(i)
                changed = True

    return improved


def _local_search(blocks, v, k, t, all_tsubsets, deadline, seed_val):
    """Local search: try to replace blocks to reduce count."""
    if not blocks:
        return blocks

    rng = random.Random(seed_val)
    improved = list(blocks)

    block_tsubs = []
    for b in improved:
        block_tsubs.append(list(combinations(b, t)))

    cover_count = Counter()
    for subs in block_tsubs:
        for sub in subs:
            cover_count[sub] += 1

    points = list(range(v))
    no_progress = 0
    max_no_progress = 10

    while time.time() < deadline and no_progress < max_no_progress:
        # Remove redundant
        redundant_found = False
        n = len(improved)
        for i in range(n):
            is_redundant = True
            for sub in block_tsubs[i]:
                if cover_count[sub] <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                for sub in block_tsubs[i]:
                    cover_count[sub] -= 1
                    if cover_count[sub] == 0:
                        del cover_count[sub]
                del improved[i]
                del block_tsubs[i]
                redundant_found = True
                break

        if redundant_found:
            no_progress = 0
            continue

        # Try merge moves
        n = len(improved)
        if n < 2:
            break

        unique_subs = [None] * n
        for i in range(n):
            unique_subs[i] = [s for s in block_tsubs[i] if cover_count[s] == 1]

        indices = list(range(n))
        rng.shuffle(indices)

        move_made = False
        for b_idx in indices:
            if time.time() > deadline:
                break
            if len(unique_subs[b_idx]) == 0:
                continue

            b_unique = unique_subs[b_idx]

            for a_idx in indices:
                if a_idx == b_idx:
                    continue
                if time.time() > deadline:
                    break

                a_unique = unique_subs[a_idx]

                union_points = set()
                for s in a_unique:
                    union_points.update(s)
                for s in b_unique:
                    union_points.update(s)

                if len(union_points) <= k:
                    a_prime = list(union_points)
                    while len(a_prime) < k:
                        p = rng.randint(0, v - 1)
                        if p not in a_prime:
                            a_prime.append(p)
                    a_prime = tuple(sorted(a_prime))

                    is_distinct = True
                    for j in range(len(improved)):
                        if j != a_idx and improved[j] == a_prime:
                            is_distinct = False
                            break
                    if not is_distinct:
                        continue

                    for sub in block_tsubs[a_idx]:
                        cover_count[sub] -= 1
                        if cover_count[sub] == 0:
                            del cover_count[sub]

                    new_tsubs = list(combinations(a_prime, t))
                    for sub in new_tsubs:
                        cover_count[sub] = cover_count.get(sub, 0) + 1

                    improved[a_idx] = a_prime
                    block_tsubs[a_idx] = new_tsubs

                    b_redundant = True
                    for sub in block_tsubs[b_idx]:
                        if cover_count.get(sub, 0) <= 1:
                            b_redundant = False
                            break

                    if b_redundant:
                        for sub in block_tsubs[b_idx]:
                            cover_count[sub] -= 1
                            if cover_count[sub] == 0:
                                del cover_count[sub]
                        del improved[b_idx]
                        del block_tsubs[b_idx]
                        move_made = True
                        break
                    else:
                        move_made = True
                        break

            if move_made:
                no_progress = 0
                break

        if not move_made:
            no_progress += 1

    improved = _remove_redundant(improved, v, k, t, deadline)
    return improved
