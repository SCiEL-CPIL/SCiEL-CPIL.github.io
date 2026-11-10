"""Covering design C(19,7,4) - bitset greedy + removal + simulated annealing."""
import time
import random
import numpy as np
from itertools import combinations

_DEADLINE = 145.0
_EXTRA = 5


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


def _block_bitset(block, v, t, idx):
    """Return a numpy bool array of length total marking which t-subsets are covered by block."""
    n = len(idx)
    bits = np.zeros(n, dtype=bool)
    for comb in combinations(block, t):
        ci = idx.get(comb, -1)
        if ci >= 0:
            bits[ci] = True
    return bits


def _greedy_construct(v, k, t, idx, total, reverse, deadline):
    """Fast greedy using a precomputed t-subset -> covering k-subsets index."""
    n = total
    uncovered = np.ones(n, dtype=bool)
    blocks = []

    all_ksubs = list(combinations(range(v), k))
    all_bitsets = []
    for ks in all_ksubs:
        bits = np.zeros(n, dtype=bool)
        for comb in combinations(ks, t):
            ci = idx.get(comb, -1)
            if ci >= 0:
                bits[ci] = True
        all_bitsets.append(bits)

    # Index: for each t-subset, list of k-subset indices that contain it
    tsub_to_ksubs = [[] for _ in range(n)]
    for i, ks in enumerate(all_ksubs):
        for comb in combinations(ks, t):
            ci = idx.get(comb, -1)
            if ci >= 0:
                tsub_to_ksubs[ci].append(i)

    while uncovered.any():
        if time.time() > deadline:
            break

        first_idx = int(np.argmax(uncovered))
        candidates = tsub_to_ksubs[first_idx]
        if not candidates:
            break

        best_i = -1
        best_gain = -1
        for ci in candidates:
            gain = int(np.sum(all_bitsets[ci] & uncovered))
            if gain > best_gain:
                best_gain = gain
                best_i = ci

        if best_i < 0:
            break

        blocks.append(list(all_ksubs[best_i]))
        uncovered &= ~all_bitsets[best_i]

    return blocks


def _improve_blocks(blocks, v, k, t, idx, deadline):
    """Deterministic removal pass."""
    n = len(idx)
    coverage = np.zeros(n, dtype=np.int32)
    for block in blocks:
        for comb in combinations(block, t):
            i = idx.get(comb, -1)
            if i >= 0:
                coverage[i] += 1

    blocks = list(blocks)
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
                if ci >= 0 and coverage[ci] <= 1:
                    removable = False
                    break
            if removable:
                for comb in combinations(block, t):
                    ci = idx.get(comb, -1)
                    if ci >= 0:
                        coverage[ci] -= 1
                blocks.pop(i)
                improved = True
                break

    return blocks


def _random_removal_pass(blocks, v, k, t, idx, deadline, n_trials=20000):
    """Randomized removal attempts."""
    n = len(idx)
    blocks = list(blocks)
    coverage = np.zeros(n, dtype=np.int32)
    for block in blocks:
        for comb in combinations(block, t):
            i = idx.get(comb, -1)
            if i >= 0:
                coverage[i] += 1

    rng = random.Random(42)
    for _ in range(n_trials):
        if time.time() > deadline:
            break
        if not blocks:
            break
        i = rng.randrange(len(blocks))
        block = blocks[i]
        removable = True
        for comb in combinations(block, t):
            ci = idx.get(comb, -1)
            if ci >= 0 and coverage[ci] <= 1:
                removable = False
                break
        if removable:
            for comb in combinations(block, t):
                ci = idx.get(comb, -1)
                if ci >= 0:
                    coverage[ci] -= 1
            blocks.pop(i)

    return blocks


def _simulated_annealing(blocks, v, k, t, idx, deadline):
    """Simulated annealing with removals and additions to escape local optima."""
    n = len(idx)
    blocks = list(blocks)
    coverage = np.zeros(n, dtype=np.int32)
    for block in blocks:
        for comb in combinations(block, t):
            i = idx.get(comb, -1)
            if i >= 0:
                coverage[i] += 1

    all_ksubs = list(combinations(range(v), k))
    all_bitsets = []
    for ks in all_ksubs:
        bits = np.zeros(n, dtype=bool)
        for comb in combinations(ks, t):
            ci = idx.get(comb, -1)
            if ci >= 0:
                bits[ci] = True
        all_bitsets.append(bits)

    rng = random.Random(123)
    start_time = time.time()
    total_time = deadline - start_time
    if total_time <= 0:
        return blocks

    current_count = len(blocks)
    best_count = current_count
    best_blocks = list(blocks)

    T_start = 5.0
    T_end = 0.01
    n_steps = int(min(120000, total_time * 400))

    for step in range(n_steps):
        if time.time() > deadline:
            break

        progress = step / max(1, n_steps)
        T = T_start * (T_end / T_start) ** progress

        if blocks and rng.random() < 0.7:
            # Removal move
            i = rng.randrange(len(blocks))
            block = blocks[i]
            uncovered_count = 0
            affected = []
            for comb in combinations(block, t):
                ci = idx.get(comb, -1)
                if ci >= 0:
                    if coverage[ci] <= 1:
                        uncovered_count += 1
                    affected.append(ci)

            if uncovered_count == 0:
                for ci in affected:
                    coverage[ci] -= 1
                blocks.pop(i)
                current_count = len(blocks)
                if current_count < best_count:
                    best_count = current_count
                    best_blocks = list(blocks)
            else:
                if rng.random() < np.exp(-uncovered_count / max(T, 1e-10)):
                    for ci in affected:
                        coverage[ci] -= 1
                    blocks.pop(i)
                    current_count = len(blocks)
        else:
            # Addition move: add a block covering many currently-uncovered t-subsets
            # (only useful if we previously over-removed; helps escape local optima)
            uncovered_mask = (coverage == 0)
            if not uncovered_mask.any():
                continue
            # Pick a random uncovered t-subset, add a block covering it
            unc_idx = np.where(uncovered_mask)[0]
            if len(unc_idx) == 0:
                continue
            pick = int(unc_idx[rng.randrange(len(unc_idx))])
            # Find a k-subset containing this t-subset with max coverage of uncovered
            # Use precomputed: iterate ksubs containing pick
            best_i = -1
            best_gain = 0
            for j, ks in enumerate(all_ksubs):
                bits = all_bitsets[j]
                if not bits[pick]:
                    continue
                gain = int(np.sum(bits & uncovered_mask))
                if gain > best_gain:
                    best_gain = gain
                    best_i = j
            if best_i >= 0:
                blocks.append(list(all_ksubs[best_i]))
                coverage += all_bitsets[best_i].astype(np.int32)
                current_count = len(blocks)

    return best_blocks


def _block_swap(blocks, v, k, t, idx, deadline):
    """Try to replace a block with another that covers the same or more uncovered t-subsets."""
    n = len(idx)
    blocks = list(blocks)
    coverage = np.zeros(n, dtype=np.int32)
    for block in blocks:
        for comb in combinations(block, t):
            i = idx.get(comb, -1)
            if i >= 0:
                coverage[i] += 1

    all_ksubs = list(combinations(range(v), k))
    all_bitsets = []
    for ks in all_ksubs:
        bits = np.zeros(n, dtype=bool)
        for comb in combinations(ks, t):
            ci = idx.get(comb, -1)
            if ci >= 0:
                bits[ci] = True
        all_bitsets.append(bits)

    # Map from frozenset to index in all_ksubs
    ksub_to_idx = {frozenset(ks): i for i, ks in enumerate(all_ksubs)}

    rng = random.Random(99)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            block = blocks[i]
            block_set = frozenset(block)
            block_idx = ksub_to_idx.get(block_set, -1)
            if block_idx < 0:
                continue
            block_bits = all_bitsets[block_idx]

            # Find t-subsets uniquely covered by this block
            unique_mask = block_bits & (coverage == 1)
            if not unique_mask.any():
                continue

            # Try to find another block that covers all unique t-subsets
            # and doesn't introduce new uncovered issues
            best_j = -1
            best_extra = -1
            for j, ks in enumerate(all_ksubs):
                if j == block_idx:
                    continue
                ks_set = set(ks)
                # Check if ks covers all unique t-subsets
                ks_bits = all_bitsets[j]
                if not np.all(unique_mask & ks_bits):
                    continue
                # Count how many currently-covered t-subsets this new block would also cover
                # (i.e., t-subsets covered by both old and new block)
                overlap = int(np.sum(block_bits & ks_bits))
                # We want to maximize overlap to minimize loss
                if overlap > best_extra:
                    best_extra = overlap
                    best_j = j

            if best_j >= 0:
                # Replace block i with all_ksubs[best_j]
                old_bits = all_bitsets[block_idx]
                new_bits = all_bitsets[best_j]
                coverage -= old_bits.astype(np.int32)
                coverage += new_bits.astype(np.int32)
                blocks[i] = list(all_ksubs[best_j])
                improved = True

    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + _DEADLINE

    # Build index and reverse map
    idx, total, reverse = _build_index(v, t)

    # Fast greedy construction
    blocks = _greedy_construct(v, k, t, idx, total, reverse, deadline)

    # Deterministic improvement pass
    if time.time() < deadline - 10:
        blocks = _improve_blocks(blocks, v, k, t, idx, deadline)

    # Randomized removal pass
    if time.time() < deadline - 8:
        blocks = _random_removal_pass(blocks, v, k, t, idx, deadline)

    # Block swap optimization
    if time.time() < deadline - 6:
        blocks = _block_swap(blocks, v, k, t, idx, deadline)

    # Simulated annealing
    if time.time() < deadline - 3:
        blocks = _simulated_annealing(blocks, v, k, t, idx, deadline)

    # Final cleanup pass
    if time.time() < deadline - 1:
        blocks = _improve_blocks(blocks, v, k, t, idx, deadline)
        blocks = _random_removal_pass(blocks, v, k, t, idx, deadline, n_trials=5000)

    # High-precision polish: repeated removal passes with increasing trial counts
    if time.time() < deadline - 0.5:
        for _ in range(3):
            if time.time() > deadline - 0.5:
                break
            blocks = _improve_blocks(blocks, v, k, t, idx, deadline)
            blocks = _random_removal_pass(blocks, v, k, t, idx, deadline, n_trials=10000)

    return blocks
