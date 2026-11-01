"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

_global_deadline = None


def _greedy_cover(v, k, t, rng=None, deadline=None):
    """Greedy set-cover: open a block on the first uncovered t-subset, then add
    the point that covers the most uncovered t-subsets until the block has k points."""
    if rng is None:
        rng = random.Random()
    if deadline is None:
        deadline = _global_deadline if _global_deadline else float('inf')
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered and time.time() < deadline:
        # Pick a random uncovered t-subset as seed (randomization helps)
        seed = rng.choice(list(uncovered))
        block = list(seed)
        block_set = set(block)
        while len(block) < k:
            # Compute gain for each candidate point
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Count how many uncovered t-subsets are completed by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    if ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x < 0:
                break
            block.append(best_x)
            block_set.add(best_x)
        block.sort()
        blocks.append(block)
        # Remove all t-subsets covered by this block
        block_set = set(block)
        for ts in itertools.combinations(block, t):
            uncovered.discard(ts)
    return blocks


def _local_search(blocks, v, k, t, deadline):
    """Try to reduce block count via local search: iteratively remove redundant
    blocks (those whose removal doesn't leave any t-subset uncovered)."""
    blocks = [list(b) for b in blocks]
    
    # Precompute coverage: for each t-subset, count how many blocks cover it
    # Use a dict mapping t-subset tuple -> count
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
            # Check if removing block bi leaves any t-subset with count 1
            redundant = True
            for ts in itertools.combinations(blocks[bi], t):
                ts = tuple(sorted(ts))
                if coverage_count.get(ts, 0) <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block bi
                removed_block = blocks.pop(bi)
                # Decrement counts for all t-subsets in removed block
                for ts in itertools.combinations(removed_block, t):
                    ts = tuple(sorted(ts))
                    coverage_count[ts] -= 1
                    if coverage_count[ts] <= 0:
                        del coverage_count[ts]
                improved = True
                break
    return blocks


def construct(v, k, t):
    global _global_deadline
    _global_deadline = time.time() + 100  # Safety margin before 120s limit
    rng = random.Random(42)

    # Phase 1: Greedy cover with randomization, multiple restarts
    best_blocks = None
    best_count = float('inf')
    
    # First greedy run
    blocks = _greedy_cover(v, k, t, rng, _global_deadline)
    best_blocks = blocks
    best_count = len(blocks)
    
    # Try more greedy restarts if time permits
    restarts = 0
    while time.time() < _global_deadline - 40 and restarts < 8:
        blocks = _greedy_cover(v, k, t, rng, _global_deadline - 40)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        restarts += 1

    # Phase 2: Local search to reduce block count
    if time.time() < _global_deadline - 5:
        best_blocks = _local_search(best_blocks, v, k, t, _global_deadline - 2)

    return best_blocks
# EVOLVE-BLOCK-END
