"""Covering design C(19,7,4) via randomized greedy + local search."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

import numpy as np


def _all_tsubsets(v, t):
    """Return list of all t-subsets of {0,...,v-1} as tuples."""
    return list(itertools.combinations(range(v), t))


def _block_covers(block, t):
    """Return list of t-subsets covered by a block."""
    return list(itertools.combinations(block, t))


def _greedy_cover(v, k, t, rng, deadline, target_subset=None):
    """
    Greedy covering with optional randomization.
    Returns list of blocks (each a sorted tuple).
    """
    all_tsubs = _all_tsubsets(v, t)
    uncovered = set(all_tsubs)
    blocks = []

    while uncovered and time.time() < deadline:
        if target_subset is not None and target_subset in uncovered:
            first = target_subset
        else:
            # Pick a random uncovered t-subset
            first = rng.choice(list(uncovered))

        block = list(first)
        rng.shuffle(block)

        # Greedily add points to maximize coverage of uncovered t-subsets
        while len(block) < k and time.time() < deadline:
            best_x = None
            best_gain = -1
            candidates = [x for x in range(v) if x not in block]
            rng.shuffle(candidates)
            for x in candidates:
                # Count how many uncovered t-subsets would be covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    if ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                elif gain == best_gain and best_x is not None:
                    # Random tie-breaking
                    if rng.random() < 0.5:
                        best_x = x
            if best_x is None:
                break
            block.append(best_x)

        block.sort()
        blocks.append(tuple(block))
        for ts in _block_covers(block, t):
            uncovered.discard(ts)

    return blocks


def _coverage_count(blocks, all_tsubs, t):
    """Count how many t-subsets are covered by the set of blocks."""
    covered = set()
    for b in blocks:
        for ts in itertools.combinations(b, t):
            covered.add(ts)
    return len(covered)


def _is_valid_cover(blocks, all_tsubs, t):
    """Check if blocks cover all t-subsets."""
    covered = set()
    for b in blocks:
        for ts in itertools.combinations(b, t):
            covered.add(ts)
    return len(covered) == len(all_tsubs)


def _local_search(blocks, v, k, t, all_tsubs, rng, deadline):
    """
    Local search: try to remove blocks that are redundant, and try swapping
    one block for a better one.
    """
    blocks = list(blocks)
    n_tsubs = len(all_tsubs)

    # Precompute for each block, the set of t-subsets it covers
    def block_tset(b):
        return frozenset(itertools.combinations(b, t))

    # Coverage count per t-subset
    from collections import Counter
    cov_count = Counter()
    for b in blocks:
        for ts in itertools.combinations(b, t):
            cov_count[ts] += 1

    # Try removing redundant blocks
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            b = blocks[i]
            # Check if all t-subsets covered by b are also covered by other blocks
            redundant = True
            for ts in itertools.combinations(b, t):
                if cov_count[ts] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                for ts in itertools.combinations(b, t):
                    cov_count[ts] -= 1
                blocks.pop(i)
                changed = True
                break  # Restart the scan after a removal

    # Try swapping: replace a block with a different block that covers more
    # uncovered or at least as many, allowing us to potentially remove another
    # This is complex; instead, try a simpler approach:
    # Try to find a block that can be replaced by a "better" block that
    # covers all the same t-subsets plus more, enabling further removals.

    # Simpler: try random swaps and see if we can reduce count
    for _ in range(100):
        if time.time() > deadline:
            break
        if len(blocks) < 2:
            break
        i = rng.randrange(len(blocks))
        b_old = blocks[i]
        # Generate a random new block
        new_block = tuple(sorted(rng.sample(range(v), k)))
        # Check if replacing b_old with new_block still covers everything
        # We need: all t-subsets covered by b_old must be covered by (blocks minus b_old) union new_block
        # Simplification: check if new_block covers at least the same t-subsets that b_old uniquely covered
        # Actually, just check validity after swap
        test_blocks = blocks[:i] + [new_block] + blocks[i+1:]
        # Quick check: does the union of t-subsets from test_blocks cover all?
        # This is expensive; do a quick check
        covered = set()
        for b in test_blocks:
            for ts in itertools.combinations(b, t):
                covered.add(ts)
        if len(covered) == n_tsubs:
            # Valid swap, but same count. Try to then remove redundant blocks
            blocks = test_blocks
            # Rebuild cov_count
            cov_count = Counter()
            for b in blocks:
                for ts in itertools.combinations(b, t):
                    cov_count[ts] += 1
            # Try removing redundant
            for j in range(len(blocks)):
                b = blocks[j]
                redundant = True
                for ts in itertools.combinations(b, t):
                    if cov_count[ts] <= 1:
                        redundant = False
                        break
                if redundant:
                    for ts in itertools.combinations(b, t):
                        cov_count[ts] -= 1
                    blocks.pop(j)
                    break

    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 20s safety margin

    all_tsubs = _all_tsubsets(v, t)
    n_tsubs = len(all_tsubs)

    best_blocks = None
    best_count = float('inf')

    # Phase 1: Multiple randomized greedy runs
    rng = random.Random(42)
    np.random.seed(42)

    run_count = 0
    while time.time() < deadline - 30:  # Reserve 30s for local search
        run_count += 1
        # Random target subset for diversity
        target = rng.choice(all_tsubs) if rng.random() < 0.3 else None
        blocks = _greedy_cover(v, k, t, rng, deadline, target_subset=target)

        if not blocks:
            continue

        # Check validity
        if not _is_valid_cover(blocks, all_tsubs, t):
            # Try to fix by adding blocks for uncovered t-subsets
            covered = set()
            for b in blocks:
                for ts in itertools.combinations(b, t):
                    covered.add(ts)
            uncovered = [ts for ts in all_tsubs if ts not in covered]
            for ts in uncovered:
                # Add a block containing this t-subset
                block = list(ts)
                remaining = [x for x in range(v) if x not in block]
                rng.shuffle(remaining)
                for x in remaining:
                    if len(block) >= k:
                        break
                    block.append(x)
                block.sort()
                blocks.append(tuple(block))

        count = len(blocks)
        if count < best_count:
            best_count = count
            best_blocks = list(blocks)

    if best_blocks is None:
        # Fallback: deterministic greedy
        blocks = _greedy_cover(v, k, t, rng, deadline)
        return [list(b) for b in blocks]

    # Phase 2: Local search on the best solution
    if time.time() < deadline - 5:
        improved = _local_search(best_blocks, v, k, t, all_tsubs, rng, deadline)
        if len(improved) < best_count:
            # Verify validity
            if _is_valid_cover(improved, all_tsubs, t):
                best_blocks = improved
                best_count = len(improved)

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
