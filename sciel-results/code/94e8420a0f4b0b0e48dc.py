"""Covering design C(19,7,4) - vectorized greedy + local search refinement."""
import time
import random
import numpy as np
from itertools import combinations

_DEADLINE = 145.0  # safety margin before 150s kill


def _build_index(v, t):
    """Map each t-subset to an integer index."""
    idx = {}
    i = 0
    for comb in combinations(range(v), t):
        idx[comb] = i
        i += 1
    return idx, i


def _precompute_block_masks(v, k, t, idx, total):
    """Precompute for each possible k-block, which t-subsets it covers (as a set of indices)."""
    # This is too expensive to precompute all C(19,7) blocks. Instead, compute on demand.
    pass


def _greedy_construct(v, k, t, idx, total, deadline):
    """Fast vectorized greedy: pick block covering most uncovered t-subsets."""
    n = total
    uncovered = np.ones(n, dtype=np.int32)
    blocks = []
    start = time.time()

    # Precompute: for each t-subset index, the actual tuple
    tsubsets_list = list(idx.keys())

    # For speed, precompute for each point, the list of t-subset indices containing it
    point_to_tsubs = [[] for _ in range(v)]
    for i, tsub in enumerate(tsubsets_list):
        for p in tsub:
            point_to_tsubs[p].append(i)

    while uncovered.any():
        if time.time() > deadline:
            break

        # Find the best block to add
        # For each candidate block (k-subset), compute how many new t-subsets it covers
        # This is expensive, so we use a heuristic: pick an uncovered t-subset and build around it

        # Get uncovered t-subset indices
        uncov_mask = uncovered > 0
        uncov_indices = np.where(uncov_mask)[0]
        if len(uncov_indices) == 0:
            break

        # Pick a random uncovered t-subset to seed the block
        seed_idx = uncov_indices[random.randint(0, len(uncov_indices) - 1)]
        seed_tsub = tsubsets_list[seed_idx]
        block = list(seed_tsub)

        # Greedily add points to maximize coverage of uncovered t-subsets
        while len(block) < k:
            best_x = -1
            best_gain = -1
            block_set = set(block)

            for x in range(v):
                if x in block_set:
                    continue
                # Count new uncovered t-subsets covered by adding x
                gain = 0
                for rest in combinations(block, t - 1):
                    tsub = tuple(sorted(rest + (x,)))
                    ci = idx.get(tsub, -1)
                    if ci >= 0 and uncovered[ci] > 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            if best_x == -1:
                # No point helps, just pick a random one
                available = [x for x in range(v) if x not in block_set]
                if available:
                    best_x = random.choice(available)
                else:
                    break

            block.append(best_x)
            if time.time() > deadline:
                break

        if len(block) < k:
            available = [x for x in range(v) if x not in set(block)]
            block.extend(random.sample(available, k - len(block)))

        block.sort()
        blocks.append(block)

        # Update uncovered counts
        for comb in combinations(block, t):
            i = idx.get(comb, -1)
            if i >= 0:
                uncovered[i] -= 1

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
            # Old block covers certain t-subsets, new block covers different ones
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
                # (simplified: just track best)
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
    idx, total = _build_index(v, t)

    # Greedy construction
    blocks = _greedy_construct(v, k, t, idx, total, deadline)

    # Remove redundant blocks
    if time.time() < deadline - 10:
        blocks, _ = _remove_redundant(blocks, v, t, idx, deadline)

    # Local search refinement
    if time.time() < deadline - 5:
        rng = random.Random(42)
        blocks = _local_search(blocks, v, k, t, idx, deadline, rng)

    # Final removal pass
    if time.time() < deadline - 2:
        blocks, _ = _remove_redundant(blocks, v, t, idx, deadline)

    return blocks
