"""Covering design C(19,7,4) - bitset greedy + simulated annealing refinement."""
import time
import random
import numpy as np
from itertools import combinations

_DEADLINE = 145.0  # safety margin before 150s kill


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


def _precompute_block_bitsets(v, k, t, idx, total):
    """Precompute bitset for each k-subset showing which t-subsets it covers."""
    all_ksubs = list(combinations(range(v), k))
    all_bitsets = []
    for ks in all_ksubs:
        bits = np.zeros(total, dtype=bool)
        for comb in combinations(ks, t):
            ci = idx.get(comb, -1)
            if ci >= 0:
                bits[ci] = True
        all_bitsets.append(bits)
    return all_ksubs, all_bitsets


def _greedy_construct(v, k, t, idx, total, deadline):
    """Fast greedy using precomputed block bitsets and numpy operations."""
    n = total
    uncovered = np.ones(n, dtype=bool)
    blocks = []
    start = time.time()

    all_ksubs, all_bitsets = _precompute_block_bitsets(v, k, t, idx, total)
    n_ksubs = len(all_ksubs)

    while uncovered.any():
        if time.time() > deadline:
            break

        # Find the k-subset that covers the most uncovered t-subsets
        # Use numpy to compute gains for all candidates at once
        gains = np.sum([int(np.sum(bs & uncovered)) for bs in all_bitsets])
        best_i = int(np.argmax(gains))
        best_gain = gains[best_i]
        
        if best_gain <= 0:
            break

        block = list(all_ksubs[best_i])
        blocks.append(block)
        uncovered &= ~all_bitsets[best_i]

    return blocks


def _compute_coverage(blocks, v, t, idx):
    """Compute coverage count for each t-subset."""
    n = len(idx)
    coverage = np.zeros(n, dtype=np.int32)
    for block in blocks:
        for comb in combinations(block, t):
            i = idx.get(comb, -1)
            if i >= 0:
                coverage[i] += 1
    return coverage


def _remove_redundant(blocks, v, t, idx, deadline):
    """Remove blocks that don't uniquely cover any t-subset."""
    n = len(idx)
    coverage = _compute_coverage(blocks, v, t, idx)
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
    return blocks, coverage


def _simulated_annealing(blocks, v, k, t, idx, deadline):
    """Simulated annealing to escape local optima in block count."""
    n = len(idx)
    blocks = list(blocks)
    coverage = _compute_coverage(blocks, v, t, idx)

    rng = random.Random(123)
    start_time = time.time()
    total_time = deadline - start_time
    if total_time <= 0:
        return blocks

    current_count = len(blocks)
    best_count = current_count
    best_blocks = list(blocks)
    best_coverage = coverage.copy()

    # Simulated annealing parameters
    T_start = 5.0
    T_end = 0.01
    n_steps = int(min(80000, total_time * 300))

    for step in range(n_steps):
        if time.time() > deadline:
            break
        if not blocks:
            break

        progress = step / max(1, n_steps)
        T = T_start * (T_end / T_start) ** progress

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
                best_coverage = coverage.copy()
        else:
            if rng.random() < np.exp(-uncovered_count / max(T, 1e-10)):
                for ci in affected:
                    coverage[ci] -= 1
                blocks.pop(i)
                current_count = len(blocks)
            else:
                pass

    return best_blocks


def _local_search(blocks, v, k, t, idx, deadline, rng):
    """Local search: try swapping points in blocks to improve coverage efficiency."""
    n = len(idx)
    blocks = list(blocks)
    coverage = _compute_coverage(blocks, v, t, idx)
    best_blocks = list(blocks)
    best_coverage = coverage.copy()
    best_len = len(blocks)

    start_time = time.time()
    max_iters = 200000

    for it in range(max_iters):
        if time.time() > deadline:
            break

        # Choose a random block and try to improve it
        if not blocks:
            break

        i = rng.randrange(len(blocks))
        old_block = list(blocks[i])

        # Try replacing one point with another
        pos = rng.randrange(k)
        old_point = old_block[pos]

        # Try all other points
        best_new_block = None
        best_gain = 0

        for new_point in range(v):
            if new_point == old_point or new_point in old_block:
                continue

            new_block = list(old_block)
            new_block[pos] = new_point
            new_block.sort()

            # Compute gain: how many t-subsets become more covered
            old_tsubs = set(idx[tuple(c)] for c in combinations(old_block, t) if tuple(c) in idx)
            new_tsubs = set(idx[tuple(c)] for c in combinations(new_block, t) if tuple(c) in idx)

            # Gain = (new t-subsets with coverage 0) - (old t-subsets with coverage 1 that become 0)
            gain = 0
            for ci in new_tsubs:
                if coverage[ci] == 0:
                    gain += 1
            for ci in old_tsubs:
                if coverage[ci] == 1 and ci not in new_tsubs:
                    gain -= 1

            if gain > best_gain:
                best_gain = gain
                best_new_block = new_block

        if best_new_block is not None and best_gain > 0:
            # Apply the swap
            old_tsubs = set(idx[tuple(c)] for c in combinations(old_block, t) if tuple(c) in idx)
            new_tsubs = set(idx[tuple(c)] for c in combinations(best_new_block, t) if tuple(c) in idx)

            for ci in old_tsubs:
                coverage[ci] -= 1
            for ci in new_tsubs:
                coverage[ci] += 1

            blocks[i] = best_new_block

            # Check if all t-subsets are still covered
            if not (coverage >= 0).all():
                # Undo
                for ci in new_tsubs:
                    coverage[ci] -= 1
                for ci in old_tsubs:
                    coverage[ci] += 1
                blocks[i] = old_block
            else:
                # Check if we can remove any blocks now
                if len(blocks) < best_len:
                    best_len = len(blocks)
                    best_blocks = list(blocks)
                    best_coverage = coverage.copy()

        # Also try removing a block if possible
        if rng.random() < 0.1:
            j = rng.randrange(len(blocks))
            block_j = blocks[j]
            removable = True
            for comb in combinations(block_j, t):
                ci = idx.get(comb, -1)
                if ci >= 0 and coverage[ci] <= 1:
                    removable = False
                    break

            if removable:
                for comb in combinations(block_j, t):
                    ci = idx.get(comb, -1)
                    if ci >= 0:
                        coverage[ci] -= 1
                blocks.pop(j)

                if len(blocks) < best_len:
                    best_len = len(blocks)
                    best_blocks = list(blocks)
                    best_coverage = coverage.copy()

    return best_blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + _DEADLINE

    # Build index
    idx, total, reverse = _build_index(v, t)

    # Greedy construction
    blocks = _greedy_construct(v, k, t, idx, total, deadline)

    # Remove redundant blocks
    if time.time() < deadline - 10:
        blocks, _ = _remove_redundant(blocks, v, t, idx, deadline)

    # Simulated annealing refinement
    if time.time() < deadline - 5:
        blocks = _simulated_annealing(blocks, v, k, t, idx, deadline)

    # Local search refinement
    if time.time() < deadline - 3:
        rng = random.Random(42)
        blocks = _local_search(blocks, v, k, t, idx, deadline, rng)

    # Final removal pass
    if time.time() < deadline - 1:
        blocks, _ = _remove_redundant(blocks, v, t, idx, deadline)

    return blocks
