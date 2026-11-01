"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def _greedy_cover(v, k, t, rng=None):
    """Greedy set-cover: open a block on the first uncovered t-subset, then add
    the point that covers the most uncovered t-subsets until the block has k points."""
    if rng is None:
        rng = random.Random()
    # Represent uncovered t-subsets as a set of tuples
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered and time.time() < _global_deadline:
        # Pick a random uncovered t-subset as seed
        seed = tuple(rng.choice(list(uncovered)))
        block = list(seed)
        block_set = set(block)
        
        while len(block) < k:
            # Find the candidate x that maximizes the number of newly covered t-subsets
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Count how many t-subsets in uncovered would be covered by adding x
                # A t-subset is covered if it contains x and t-1 elements from block
                # We need to count tuples in uncovered that contain x and (t-1) from block
                gain = 0
                # For efficiency, we can iterate over combinations of block of size t-1
                # and check if (that combo + x) is in uncovered
                if t - 1 <= len(block):
                    for combo in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(combo + (x,)))
                        if ts in uncovered:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                # Should not happen if uncovered is not empty and we haven't covered all
                break
            block.append(best_x)
            block_set.add(best_x)
        
        block.sort()
        blocks.append(block)
        # Remove all t-subsets covered by this block from uncovered
        for ts in itertools.combinations(block, t):
            ts = tuple(sorted(ts))
            if ts in uncovered:
                uncovered.remove(ts)
    return blocks


def _local_search(blocks, v, k, t, deadline):
    """Try to reduce block count via local search: remove a block if it's redundant."""
    blocks = [list(b) for b in blocks]
    
    # Precompute coverage: for each t-subset, how many blocks cover it
    # We use a dictionary mapping t-subset tuple to count
    coverage_count = {}
    for block in blocks:
        for ts in itertools.combinations(block, t):
            ts = tuple(sorted(ts))
            coverage_count[ts] = coverage_count.get(ts, 0) + 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                break
            # Check if block bi is redundant (all its t-subsets are covered by other blocks)
            redundant = True
            for ts in itertools.combinations(blocks[bi], t):
                ts = tuple(sorted(ts))
                if coverage_count.get(ts, 0) <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block bi
                # Decrement coverage counts for its t-subsets
                for ts in itertools.combinations(blocks[bi], t):
                    ts = tuple(sorted(ts))
                    coverage_count[ts] -= 1
                    if coverage_count[ts] == 0:
                        del coverage_count[ts]
                blocks.pop(bi)
                improved = True
                break
    return blocks


def construct(v, k, t):
    global _global_deadline
    _global_deadline = time.time() + 110  # Safety margin before 120s limit
    rng = random.Random(42)

    # Phase 1: Greedy cover
    blocks = _greedy_cover(v, k, t, rng)
    best_blocks = blocks
    best_count = len(blocks)
    
    # Try a few more greedy restarts if time permits
    restarts = 0
    while time.time() < _global_deadline - 30 and restarts < 5:
        blocks = _greedy_cover(v, k, t, rng)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        restarts += 1

    # Phase 2: Local search to reduce block count
    if time.time() < _global_deadline - 5:
        best_blocks = _local_search(best_blocks, v, k, t, _global_deadline - 2)

    return best_blocks
# EVOLVE-BLOCK-END
