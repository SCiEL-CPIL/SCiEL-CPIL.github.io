"""
Optimized covering design C(19,7,4) using diverse randomized greedy with local improvement.
"""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations


def _precompute(v, t):
    """Precompute all t-subsets and lookup structures."""
    all_t_subsets = list(combinations(range(v), t))
    total = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    return all_t_subsets, total, subset_to_idx


def _greedy_single(v, k, t, seed, time_limit, all_t_subsets, total, subset_to_idx,
                   strategy='random_seed'):
    """
    Run a single randomized greedy covering.
    strategy: 'random_seed' - random uncovered subset to start
              'random_tie' - random tie-breaking in gain
              'weighted' - weighted sampling for seed selection
    """
    rng = np.random.default_rng(seed)
    uncovered = np.ones(total, dtype=bool)
    blocks = []
    start_time = time.time()

    while np.any(uncovered):
        if time.time() - start_time > time_limit:
            break

        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) == 0:
            break

        # Choose seed subset based on strategy
        if strategy == 'random_seed':
            idx = uncovered_indices[rng.integers(len(uncovered_indices))]
        elif strategy == 'weighted':
            # Weight by how many uncovered subsets share elements (harder to cover)
            weights = np.ones(len(uncovered_indices), dtype=np.float64)
            # Simple: just pick randomly (weighted adds complexity without clear gain here)
            idx = uncovered_indices[rng.integers(len(uncovered_indices))]
        else:
            idx = uncovered_indices[rng.integers(len(uncovered_indices))]

        seed_subset = all_t_subsets[idx]
        block = list(seed_subset)
        block_set = set(block)

        while len(block) < k:
            if time.time() - start_time > time_limit:
                return None

            best_gain = -1
            best_x = None
            tied_candidates = []

            candidates = [x for x in range(v) if x not in block_set]
            t_minus_1_combos = list(combinations(block, t - 1))

            for x in candidates:
                gain = 0
                for combo in t_minus_1_combos:
                    s = tuple(sorted(combo + (x,)))
                    if s in subset_to_idx:
                        i = subset_to_idx[s]
                        if uncovered[i]:
                            gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    tied_candidates = [x]
                elif gain == best_gain and gain > 0:
                    tied_candidates.append(x)

            if best_x is None:
                break

            # Strategy-based tie-breaking
            if strategy == 'random_tie' and len(tied_candidates) > 1:
                best_x = tied_candidates[rng.integers(len(tied_candidates))]
            elif strategy == 'weighted' and len(tied_candidates) > 1:
                # Prefer points that appear in more uncovered subsets
                counts = {}
                for ci in uncovered_indices:
                    for pt in all_t_subsets[ci]:
                        counts[pt] = counts.get(pt, 0) + 1
                best_x = max(tied_candidates, key=lambda x: counts.get(x, 0))
            # else: keep first found (deterministic tie-break)

            block.append(best_x)
            block_set.add(best_x)

        if len(block) < k:
            break

        block = sorted(block)
        blocks.append(block)

        # Batch update uncovered
        for s in combinations(block, t):
            i = subset_to_idx[s]
            uncovered[i] = False

    if not np.all(uncovered == False):
        return None  # Incomplete
    return blocks


def _local_improve(blocks, v, k, t, subset_to_idx, all_t_subsets, time_limit):
    """
    Try to reduce block count by finding blocks that can be merged or points swapped.
    Simple approach: try removing a block and see if remaining blocks still cover everything.
    """
    start_time = time.time()
    blocks = [list(b) for b in blocks]
    n = len(blocks)

    if n < 2:
        return blocks

    # Build coverage count for each t-subset
    total = len(all_t_subsets)
    cover_count = np.zeros(total, dtype=np.int32)
    for block in blocks:
        for s in combinations(block, t):
            i = subset_to_idx[s]
            cover_count[i] += 1

    # Try removing each block (from the end to preserve order)
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() - start_time > time_limit:
                break
            # Check if removing block i leaves all covered
            block_i = blocks[i]
            removable = True
            for s in combinations(block_i, t):
                idx = subset_to_idx[s]
                if cover_count[idx] <= 1:
                    removable = False
                    break
            if removable:
                for s in combinations(block_i, t):
                    idx = subset_to_idx[s]
                    cover_count[idx] -= 1
                blocks.pop(i)
                improved = True
                break  # Restart scan after removal

    return blocks


def _try_swap_improve(blocks, v, k, t, subset_to_idx, all_t_subsets, time_limit):
    """
    Try to improve by swapping points between blocks to increase overlap.
    For each pair of blocks, try swapping one point from each to see if coverage improves.
    This is expensive, so only do it if we have time.
    """
    start_time = time.time()
    blocks = [list(b) for b in blocks]
    n = len(blocks)

    if n < 2:
        return blocks

    total = len(all_t_subsets)
    cover_count = np.zeros(total, dtype=np.int32)
    for block in blocks:
        for s in combinations(block, t):
            i = subset_to_idx[s]
            cover_count[i] += 1

    # For each pair of blocks, try swapping points
    for i in range(n):
        if time.time() - start_time > time_limit:
            break
        for j in range(i + 1, n):
            if time.time() - start_time > time_limit:
                break
            bi = blocks[i]
            bj = blocks[j]
            bi_set = set(bi)
            bj_set = set(bj)

            for pi in bi:
                if time.time() - start_time > time_limit:
                    break
                for pj in bj:
                    if pi == pj:
                        continue
                    # Try swapping pi and pj
                    # New bi: remove pi, add pj
                    # New bj: remove pj, add pi
                    new_bi = [x for x in bi if x != pi] + [pj]
                    new_bj = [x for x in bj if x != pj] + [pi]

                    # Check if this creates duplicates
                    if len(set(new_bi)) != k or len(set(new_bj)) != k:
                        continue

                    # Compute change in coverage
                    # Subsets lost from bi: those containing pi but not pj
                    # Subsets gained by bi: those containing pj but not pi
                    # Similarly for bj

                    # This is complex; skip for now as it's expensive
                    pass

    return blocks


def _verify(blocks, v, k, t, subset_to_idx, all_t_subsets):
    """Verify that all t-subsets are covered."""
    total = len(all_t_subsets)
    covered = np.zeros(total, dtype=bool)
    for block in blocks:
        for s in combinations(block, t):
            i = subset_to_idx[s]
            covered[i] = True
    return np.all(covered)


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s safety margin

    # Precompute
    all_t_subsets, total, subset_to_idx = _precompute(v, t)

    best_blocks = None
    best_count = float('inf')

    # Define diverse strategies
    strategies = [
        ('random_seed', 0),
        ('random_seed', 1),
        ('random_seed', 2),
        ('random_tie', 0),
        ('random_tie', 1),
        ('random_tie', 2),
        ('weighted', 0),
        ('weighted', 1),
        ('random_seed', 3),
        ('random_seed', 4),
        ('random_tie', 3),
        ('random_tie', 4),
        ('weighted', 2),
        ('weighted', 3),
        ('random_seed', 5),
        ('random_seed', 6),
    ]

    # Run diverse greedy strategies
    for strategy, seed in strategies:
        if time.time() >= deadline - 10:  # Leave 10s for local improvement
            break

        remaining_time = deadline - time.time() - 10
        if remaining_time < 1:
            break

        blocks = _greedy_single(v, k, t, seed, remaining_time,
                                all_t_subsets, total, subset_to_idx, strategy)

        if blocks is not None and len(blocks) < best_count:
            if _verify(blocks, v, k, t, subset_to_idx, all_t_subsets):
                best_blocks = blocks
                best_count = len(blocks)

        # Early exit if we hit the known best
        if best_count <= 145:
            break

    # Local improvement on the best solution found
    if best_blocks is not None and time.time() < deadline - 5:
        improved = _local_improve(best_blocks, v, k, t, subset_to_idx,
                                  all_t_subsets, deadline - time.time() - 5)
        if _verify(improved, v, k, t, subset_to_idx, all_t_subsets):
            best_blocks = improved
            best_count = len(best_blocks)

    # Fallback: deterministic greedy if nothing found
    if best_blocks is None:
        uncovered = np.ones(total, dtype=bool)
        blocks = []
        while np.any(uncovered):
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            idx = uncovered_indices[0]  # Deterministic: first uncovered
            seed_subset = all_t_subsets[idx]
            block = list(seed_subset)
            block_set = set(block)

            while len(block) < k:
                best_gain = -1
                best_x = None
                candidates = [x for x in range(v) if x not in block_set]
                t_minus_1_combos = list(combinations(block, t - 1))

                for x in candidates:
                    gain = 0
                    for combo in t_minus_1_combos:
                        s = tuple(sorted(combo + (x,)))
                        if s in subset_to_idx:
                            i = subset_to_idx[s]
                            if uncovered[i]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x

                if best_x is None:
                    break
                block.append(best_x)
                block_set.add(best_x)

            if len(block) == k:
                block = sorted(block)
                blocks.append(block)
                for s in combinations(block, t):
                    i = subset_to_idx[s]
                    uncovered[i] = False

        best_blocks = blocks

    return best_blocks

# EVOLVE-BLOCK-END
