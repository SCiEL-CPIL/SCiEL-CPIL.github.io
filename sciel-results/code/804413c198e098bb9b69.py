"""Covering design C(19,7,4) - bitset greedy + vectorized local search + SA."""
import time
import random
import numpy as np
from itertools import combinations

_DEADLINE = 145.0


def _build_index(v, t):
    """Map each t-subset to an integer index. Return (idx_dict, total, reverse_list)."""
    idx = {}
    reverse = []
    i = 0
    for comb in combinations(range(v), t):
        idx[comb] = i
        reverse.append(comb)
        i += 1
    return idx, i, reverse


def _precompute_block_masks(v, k, t, idx, total):
    """Precompute for every k-subset a numpy bool mask of which t-subsets it covers."""
    all_ksubs = list(combinations(range(v), k))
    masks = np.zeros((len(all_ksubs), total), dtype=np.int8)
    for i, ks in enumerate(all_ksubs):
        for comb in combinations(ks, t):
            ci = idx.get(comb, -1)
            if ci >= 0:
                masks[i, ci] = 1
    return all_ksubs, masks


def _greedy_construct(v, k, t, idx, total, all_ksubs, masks, deadline):
    """Greedy set-cover: pick k-subset covering most still-uncovered t-subsets."""
    n = total
    uncovered = np.ones(n, dtype=np.int8)
    blocks = []
    remaining = np.ones(len(all_ksubs), dtype=bool)

    while uncovered.any():
        if time.time() > deadline:
            break

        # Gain for each remaining candidate = number of uncovered t-subsets it covers
        gains = masks[remaining].dot(uncovered)
        if gains.max() <= 0:
            break
        best_local = int(np.argmax(gains))
        best_global = int(np.where(remaining)[0][best_local])

        blocks.append(list(all_ksubs[best_global]))
        uncovered = np.maximum(uncovered - masks[best_global], 0)
        remaining[best_global] = False

    return blocks


def _coverage_vector(blocks, v, t, idx):
    """Compute coverage count vector."""
    n = len(idx)
    cov = np.zeros(n, dtype=np.int32)
    for block in blocks:
        for comb in combinations(block, t):
            ci = idx.get(comb, -1)
            if ci >= 0:
                cov[ci] += 1
    return cov


def _remove_redundant(blocks, v, t, idx, cov, deadline):
    """Remove blocks that are fully redundant (all their t-subsets covered by others)."""
    blocks = list(blocks)
    n = len(idx)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            block = blocks[i]
            removable = True
            for comb in combinations(block, t):
                ci = idx.get(comb, -1)
                if ci >= 0 and cov[ci] <= 1:
                    removable = False
                    break
            if removable:
                for comb in combinations(block, t):
                    ci = idx.get(comb, -1)
                    if ci >= 0:
                        cov[ci] -= 1
                blocks.pop(i)
                improved = True
                break
    return blocks, cov


def _simulated_annealing(blocks, v, k, t, idx, all_ksubs, masks, deadline):
    """SA with vectorized coverage updates and redundant-block removal after each move."""
    n = len(idx)
    blocks = list(blocks)
    cov = _coverage_vector(blocks, v, t, idx)

    rng = random.Random(123)
    start_time = time.time()
    total_time = deadline - start_time
    if total_time <= 0:
        return blocks

    best_count = len(blocks)
    best_blocks = list(blocks)
    best_cov = cov.copy()

    T_start = 8.0
    T_end = 0.005
    n_steps = int(min(120000, total_time * 400))

    for step in range(n_steps):
        if time.time() > deadline:
            break
        if not blocks:
            break

        progress = step / max(1, n_steps)
        T = T_start * (T_end / T_start) ** progress

        # Randomly try to remove a block
        i = rng.randrange(len(blocks))
        block = blocks[i]

        # Compute which t-subsets become uncovered if we remove this block
        affected = []
        uncovered_count = 0
        for comb in combinations(block, t):
            ci = idx.get(comb, -1)
            if ci >= 0:
                if cov[ci] <= 1:
                    uncovered_count += 1
                affected.append(ci)

        if uncovered_count == 0:
            # Safe removal
            for ci in affected:
                cov[ci] -= 1
            blocks.pop(i)
            # Remove redundancies
            blocks, cov = _remove_redundant(blocks, v, t, idx, cov, deadline)
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = list(blocks)
                best_cov = cov.copy()
        else:
            # Accept with probability exp(-uncovered_count / T)
            if rng.random() < np.exp(-uncovered_count / max(T, 1e-10)):
                for ci in affected:
                    cov[ci] -= 1
                blocks.pop(i)
                # After accepting a bad move, try to fix by adding a good block
                if cov.min() < 0:
                    # Find uncovered t-subsets and add a block that covers many of them
                    uncov_mask = (cov <= 0).astype(np.int8)
                    if uncov_mask.any():
                        gains = masks.dot(uncov_mask)
                        # Exclude blocks already in the set
                        in_set = set(tuple(b) for b in blocks)
                        for gi in np.argsort(-gains):
                            if tuple(all_ksubs[gi]) not in in_set:
                                blocks.append(list(all_ksubs[gi]))
                                cov += masks[gi]
                                break
                # Remove redundancies
                blocks, cov = _remove_redundant(blocks, v, t, idx, cov, deadline)

    return best_blocks


def _block_swap(blocks, v, k, t, idx, all_ksubs, masks, deadline, n_trials=50000):
    """Try to replace a block with another that maintains coverage with fewer or equal blocks."""
    blocks = list(blocks)
    cov = _coverage_vector(blocks, v, t, idx)
    n = len(idx)
    rng = random.Random(77)

    for _ in range(n_trials):
        if time.time() > deadline:
            break
        if len(blocks) < 2:
            break
        i = rng.randrange(len(blocks))
        block = blocks[i]

        # Find t-subsets that would become uncovered
        critical = []
        for comb in combinations(block, t):
            ci = idx.get(comb, -1)
            if ci >= 0 and cov[ci] <= 1:
                critical.append(ci)

        if not critical:
            # Block is redundant, remove it
            for comb in combinations(block, t):
                ci = idx.get(comb, -1)
                if ci >= 0:
                    cov[ci] -= 1
            blocks.pop(i)
            continue

        # Find a replacement block that covers all critical t-subsets
        critical_mask = np.zeros(n, dtype=np.int8)
        for ci in critical:
            critical_mask[ci] = 1

        # Need a block that covers all critical t-subsets
        # Compute for each candidate how many critical it covers
        coverage_of_critical = masks.dot(critical_mask)
        need = len(critical)
        candidates = np.where(coverage_of_critical == need)[0]
        if len(candidates) == 0:
            continue

        # Pick a candidate not already in the set
        in_set = set(tuple(b) for b in blocks)
        chosen = -1
        for ci in candidates:
            if tuple(all_ksubs[ci]) not in in_set:
                chosen = ci
                break
        if chosen < 0:
            continue

        # Replace
        for comb in combinations(block, t):
            ci = idx.get(comb, -1)
            if ci >= 0:
                cov[ci] -= 1
        blocks.pop(i)
        blocks.append(list(all_ksubs[chosen]))
        cov += masks[chosen]

        # Remove redundancies
        blocks, cov = _remove_redundant(blocks, v, t, idx, cov, deadline)

    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + _DEADLINE

    # Build index
    idx, total, reverse = _build_index(v, t)

    # Precompute masks for all k-subsets
    all_ksubs, masks = _precompute_block_masks(v, k, t, idx, total)

    # Greedy construction
    blocks = _greedy_construct(v, k, t, idx, total, all_ksubs, masks, deadline)

    # Initial redundancy removal
    cov = _coverage_vector(blocks, v, t, idx)
    blocks, cov = _remove_redundant(blocks, v, t, idx, cov, deadline)

    # Block swap phase
    if time.time() < deadline - 10:
        blocks = _block_swap(blocks, v, k, t, idx, all_ksubs, masks, deadline)
        cov = _coverage_vector(blocks, v, t, idx)
        blocks, cov = _remove_redundant(blocks, v, t, idx, cov, deadline)

    # Simulated annealing
    if time.time() < deadline - 3:
        blocks = _simulated_annealing(blocks, v, k, t, idx, all_ksubs, masks, deadline)

    # Final redundancy removal
    cov = _coverage_vector(blocks, v, t, idx)
    blocks, cov = _remove_redundant(blocks, v, t, idx, cov, deadline)

    return blocks
