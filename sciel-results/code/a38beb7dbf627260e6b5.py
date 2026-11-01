"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def _greedy_cover(v, k, t, rng=None, deadline=None):
    """Greedy set-cover with efficient gain computation."""
    if rng is None:
        rng = random.Random()
    if deadline is None:
        deadline = time.time() + 100
    
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    # Precompute all t-sets as a list for fast random selection
    all_tsets = list(uncovered)
    # Map from index to tset for O(1) random pick without rebuilding list
    # We'll maintain a list and swap-remove when covering
    
    while uncovered and time.time() < deadline:
        # Pick a random uncovered t-subset as seed using index
        idx = rng.randrange(len(all_tsets))
        seed = all_tsets[idx]
        
        # Remove seed from active list (swap with last)
        all_tsets[idx] = all_tsets[-1]
        all_tsets.pop()
        uncovered.discard(seed)
        
        block = list(seed)
        block_set = set(block)
        
        while len(block) < k:
            # Find the candidate x that maximizes the number of newly covered t-subsets
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                if t - 1 <= len(block):
                    for combo in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(combo + (x,)))
                        if ts in uncovered:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            block_set.add(best_x)
        
        block.sort()
        blocks.append(block)
        # Remove all t-subsets covered by this block from uncovered
        for ts in itertools.combinations(block, t):
            ts = tuple(sorted(ts))
            if ts in uncovered:
                uncovered.discard(ts)
                # Also remove from all_tsets if present
                # This is O(n) but we can defer cleanup
    return blocks


def _local_search(blocks, v, k, t, deadline):
    """Try to reduce block count via local search: remove redundant blocks."""
    blocks = [list(b) for b in blocks]
    
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
            redundant = True
            for ts in itertools.combinations(blocks[bi], t):
                ts = tuple(sorted(ts))
                if coverage_count.get(ts, 0) <= 1:
                    redundant = False
                    break
            if redundant:
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
    _global_deadline = time.time() + 115  # Safety margin before 120s limit
    rng = random.Random(42)

    # Phase 1: Greedy cover
    blocks = _greedy_cover(v, k, t, rng, _global_deadline - 10)
    best_blocks = blocks
    best_count = len(blocks)
    
    # Try more greedy restarts if time permits
    restarts = 0
    while time.time() < _global_deadline - 15 and restarts < 10:
        blocks = _greedy_cover(v, k, t, rng, _global_deadline - 10)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        restarts += 1

    # Phase 2: Local search to reduce block count
    if time.time() < _global_deadline - 5:
        best_blocks = _local_search(best_blocks, v, k, t, _global_deadline - 2)

    return best_blocks
# EVOLVE-BLOCK-END
