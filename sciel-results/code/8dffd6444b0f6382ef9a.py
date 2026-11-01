"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys
import numpy as np

_global_deadline = None


def _greedy_cover(v, k, t, rng=None, deadline=None):
    """Greedy set-cover with vectorized gain computation."""
    if rng is None:
        rng = random.Random()
    if deadline is None:
        deadline = _global_deadline if _global_deadline else float('inf')
    
    # Precompute all t-subsets of the current block incrementally
    # Use a set for O(1) lookup
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered and time.time() < deadline:
        # Pick a random uncovered t-subset as seed
        seed = rng.choice(list(uncovered))
        block = list(seed)
        block_set = set(block)
        
        while len(block) < k:
            if time.time() >= deadline:
                break
            # Compute gain for each candidate point
            # A point x covers t-subsets that contain x and t-1 elements from block
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Count uncovered t-subsets that would be completed by adding x
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
        for ts in itertools.combinations(block, t):
            uncovered.discard(ts)
    
    return blocks


def _build_coverage(blocks, v, k, t):
    """Build coverage count dict: t-subset tuple -> number of blocks covering it."""
    coverage_count = {}
    for block in blocks:
        for ts in itertools.combinations(block, t):
            ts = tuple(sorted(ts))
            coverage_count[ts] = coverage_count.get(ts, 0) + 1
    return coverage_count


def _remove_redundant(blocks, v, k, t, deadline):
    """Iteratively remove redundant blocks (those with no uniquely-covered t-subset)."""
    blocks = [list(b) for b in blocks]
    coverage_count = _build_coverage(blocks, v, k, t)

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
                removed_block = blocks.pop(bi)
                for ts in itertools.combinations(removed_block, t):
                    ts = tuple(sorted(ts))
                    coverage_count[ts] -= 1
                    if coverage_count[ts] <= 0:
                        del coverage_count[ts]
                improved = True
                break
    return blocks, coverage_count


def _local_search(blocks, v, k, t, deadline):
    """Local search: iteratively try to remove or replace blocks to reduce count.
    Uses coverage tracking to identify critical blocks and attempt swaps."""
    blocks = [list(b) for b in blocks]
    coverage_count = _build_coverage(blocks, v, k, t)
    rng = random.Random(42)
    
    n_tsubs_per_block = int(np.prod([k - i for i in range(k - t + 1)])) if k >= t else 0
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(blocks)
        
        # Compute uniqueness score for each block
        block_uniqueness = []
        for bi in range(n):
            unique_count = 0
            for ts in itertools.combinations(blocks[bi], t):
                ts = tuple(sorted(ts))
                if coverage_count.get(ts, 0) == 1:
                    unique_count += 1
            block_uniqueness.append(unique_count)
        
        # Try removing blocks with 0 unique coverage
        for bi in range(n):
            if time.time() >= deadline:
                break
            if block_uniqueness[bi] == 0:
                removed_block = blocks.pop(bi)
                block_uniqueness.pop(bi)
                for ts in itertools.combinations(removed_block, t):
                    ts = tuple(sorted(ts))
                    coverage_count[ts] -= 1
                    if coverage_count[ts] <= 0:
                        del coverage_count[ts]
                improved = True
                break
        if improved:
            continue
        
        # Try replacing a low-uniqueness block with a better one
        # Sort by uniqueness (ascending) - try to improve the worst blocks
        order = sorted(range(len(blocks)), key=lambda i: block_uniqueness[i])
        
        tried = 0
        for bi in order:
            if time.time() >= deadline or tried >= 10:
                break
            if block_uniqueness[bi] > 5:
                break  # Too unique to safely modify
            
            tried += 1
            orig_block = blocks[bi]
            orig_set = set(orig_block)
            
            # Try several random replacements
            for attempt in range(15):
                if time.time() >= deadline:
                    break
                
                # Create a candidate block: keep some original points, add random
                keep_count = min(3, len(orig_block))
                kept = rng.sample(orig_block, keep_count)
                remaining_points = [x for x in range(v) if x not in kept]
                if len(remaining_points) < k - keep_count:
                    continue
                new_block = kept + rng.sample(remaining_points, k - keep_count)
                new_block.sort()
                
                if new_block == orig_block:
                    continue
                
                # Check if replacing bi with new_block maintains full coverage
                # Only need to verify t-subsets that were uniquely covered by block bi
                # or are now potentially uncovered
                test_blocks = [list(b) for b in blocks if b != orig_block]
                test_blocks.append(new_block)
                
                # Quick check: verify all t-subsets of orig_block are still covered
                valid = True
                for ts in itertools.combinations(orig_block, t):
                    ts = tuple(sorted(ts))
                    # Check if any block in test_blocks covers ts
                    found = False
                    for tb in test_blocks:
                        if set(ts).issubset(set(tb)):
                            found = True
                            break
                    if not found:
                        valid = False
                        break
                
                if valid:
                    # Accept the replacement
                    blocks[bi] = new_block
                    coverage_count = _build_coverage(blocks, v, k, t)
                    improved = True
                    break
            
            if improved:
                break
    
    return blocks


def _merge_pass(blocks, v, k, t, deadline):
    """Try to merge two blocks into one if their union fits in k points
    and the merged block covers all t-subsets of both."""
    blocks = [list(b) for b in blocks]
    coverage_count = _build_coverage(blocks, v, k, t)
    rng = random.Random(99)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(blocks)
        
        # Try pairs of blocks
        for i in range(min(n, 100)):
            if time.time() >= deadline:
                break
            for j in range(i + 1, min(n, 100)):
                if time.time() >= deadline:
                    break
                
                set_i = set(blocks[i])
                set_j = set(blocks[j])
                union = set_i | set_j
                if len(union) > k:
                    continue
                
                # Create merged block
                merged = sorted(union)
                if len(merged) < k:
                    remaining = set(range(v)) - set(merged)
                    extra = rng.sample(sorted(remaining), k - len(merged))
                    merged = sorted(merged + extra)
                
                # Check if merged block covers all t-subsets of blocks[i] and blocks[j]
                # and that removing both and adding merged maintains full coverage
                test_blocks = [list(b) for b in blocks if b != blocks[i] and b != blocks[j]]
                test_blocks.append(merged)
                
                # Verify coverage of all t-subsets from blocks[i] and blocks[j]
                valid = True
                for ts in itertools.combinations(blocks[i], t):
                    ts = tuple(sorted(ts))
                    if not set(ts).issubset(set(merged)):
                        # Need to check if another block covers it
                        found = False
                        for tb in test_blocks:
                            if tb != merged and set(ts).issubset(set(tb)):
                                found = True
                                break
                        if not found:
                            valid = False
                            break
                if valid:
                    for ts in itertools.combinations(blocks[j], t):
                        ts = tuple(sorted(ts))
                        if not set(ts).issubset(set(merged)):
                            found = False
                            for tb in test_blocks:
                                if tb != merged and set(ts).issubset(set(tb)):
                                    found = True
                                    break
                            if not found:
                                valid = False
                                break
                
                if valid and len(test_blocks) < len(blocks):
                    blocks = test_blocks
                    coverage_count = _build_coverage(blocks, v, k, t)
                    improved = True
                    break
            if improved:
                break
    
    return blocks


def construct(v, k, t):
    global _global_deadline
    _global_deadline = time.time() + 115  # Safety margin before 120s limit
    rng = random.Random(42)

    # Phase 1: Greedy cover with multiple restarts
    best_blocks = None
    best_count = float('inf')

    # First greedy run
    blocks = _greedy_cover(v, k, t, rng, _global_deadline)
    best_blocks = blocks
    best_count = len(blocks)

    # Try more greedy restarts if time permits
    restarts = 0
    while time.time() < _global_deadline - 70 and restarts < 4:
        blocks = _greedy_cover(v, k, t, rng, _global_deadline - 70)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        restarts += 1

    # Phase 2: Remove redundant blocks
    if time.time() < _global_deadline - 30:
        best_blocks, _ = _remove_redundant(best_blocks, v, k, t, _global_deadline - 30)

    # Phase 3: Local search (swap/replacement)
    if time.time() < _global_deadline - 15:
        best_blocks = _local_search(best_blocks, v, k, t, _global_deadline - 15)

    # Phase 4: Merge pass
    if time.time() < _global_deadline - 8:
        best_blocks = _merge_pass(best_blocks, v, k, t, _global_deadline - 8)

    # Phase 5: Final redundancy removal
    if time.time() < _global_deadline - 2:
        best_blocks, _ = _remove_redundant(best_blocks, v, k, t, _global_deadline - 2)

    return best_blocks
# EVOLVE-BLOCK-END
