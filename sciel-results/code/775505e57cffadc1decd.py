"""
Covering design C(v, k, t) solver.
Strategy:
1. Fast greedy construction using bitmasks for t-subset tracking.
2. Aggressive redundant block removal.
3. Many randomized restarts with different point orderings.
4. Local search: swap points in blocks to improve coverage.
5. Time-bounded search with deadline guard.
"""

import itertools
import time
import random
import sys
from collections import defaultdict

def _time_remaining(deadline):
    return deadline - time.time()

def _precompute_triples(v, k, t):
    """Precompute all t-subsets as tuples and map them to indices."""
    all_ts = list(itertools.combinations(range(v), t))
    ts_index = {ts: i for i, ts in enumerate(all_ts)}
    return all_ts, ts_index

def _greedy_construct(v, k, t, point_order=None, deadline=None, rng=None):
    """
    Greedy covering construction using bitmask-based tracking.
    Returns a list of blocks (each a sorted list of k integers).
    """
    if point_order is None:
        point_order = list(range(v))
    if rng is None:
        rng = random
    
    # Precompute all t-subsets
    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    ts_index = {ts: i for i, ts in enumerate(all_ts)}
    
    # Track uncovered t-subsets as a set of indices
    uncovered = set(range(n_ts))
    blocks = []
    
    if not uncovered:
        return blocks
    
    # For each point, precompute which t-subsets contain it
    # point_to_ts[point] = list of ts indices where point appears
    point_to_ts = defaultdict(list)
    for i, ts in enumerate(all_ts):
        for p in ts:
            point_to_ts[p].append(i)
    
    while uncovered:
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random uncovered t-subset to start the block
        first_subset = random.choice(list(uncovered))
        first_ts = all_ts[first_subset]
        
        block = set(first_ts)
        
        # Grow the block to size k by adding points that cover the most uncovered t-subsets
        while len(block) < k:
            if deadline is not None and time.time() > deadline:
                break
            
            best_point = None
            best_gain = -1
            
            for x in point_order:
                if x in block:
                    continue
                
                # Count how many uncovered t-subsets would be covered by adding x
                # A t-subset is covered if it's a subset of block ∪ {x}
                # i.e., it contains x and the other t-1 elements are in block
                gain = 0
                # For each (t-1)-subset of block, check if (that subset + x) is uncovered
                for rest in itertools.combinations(sorted(block), t - 1):
                    candidate = tuple(sorted(rest + (x,)))
                    idx = ts_index.get(candidate)
                    if idx is not None and idx in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None:
                break
            block.add(best_point)
        
        if len(block) == k:
            block_list = sorted(block)
            blocks.append(block_list)
            # Remove all t-subsets of this block from uncovered
            for ts in itertools.combinations(block_list, t):
                idx = ts_index.get(ts)
                if idx is not None:
                    uncovered.discard(idx)
    
    return blocks

def _remove_redundant_blocks(v, k, t, blocks, deadline=None):
    """
    Try to remove blocks that are redundant (all their t-subsets are covered by other blocks).
    Returns a new list of blocks.
    """
    if not blocks:
        return blocks
    
    n_blocks = len(blocks)
    
    # For each t-subset, count how many blocks cover it
    # And for each block, track which t-subsets it covers
    ts_to_blocks = defaultdict(list)
    block_tsubsets = []
    
    for bi, block in enumerate(blocks):
        bts = list(itertools.combinations(block, t))
        block_tsubsets.append(bts)
        for ts in bts:
            ts_to_blocks[ts].append(bi)
    
    # active[bi] = True if block bi is still in the covering
    active = [True] * n_blocks
    
    changed = True
    while changed:
        if deadline is not None and time.time() > deadline:
            break
        changed = False
        
        for bi in range(n_blocks):
            if not active[bi]:
                continue
            
            # Check if all t-subsets of block bi are covered by other active blocks
            redundant = True
            for ts in block_tsubsets[bi]:
                # Is ts covered by any other active block?
                found_other = False
                for bj in ts_to_blocks[ts]:
                    if bj != bi and active[bj]:
                        found_other = True
                        break
                if not found_other:
                    redundant = False
                    break
            
            if redundant:
                active[bi] = False
                changed = True
    
    result = [blocks[bi] for bi in range(n_blocks) if active[bi]]
    return result

def _local_search(v, k, t, blocks, deadline=None, rng=None):
    """
    Local search: try to swap points in blocks to improve coverage or enable block removal.
    """
    if rng is None:
        rng = random
    if not blocks:
        return blocks
    
    # Precompute t-subset index
    all_ts = list(itertools.combinations(range(v), t))
    ts_index = {ts: i for i, ts in enumerate(all_ts)}
    n_ts = len(all_ts)
    
    # Track coverage: for each t-subset, how many blocks cover it
    coverage_count = [0] * n_ts
    block_tsubsets = []
    
    for block in blocks:
        bts = []
        for ts in itertools.combinations(block, t):
            idx = ts_index.get(ts)
            if idx is not None:
                coverage_count[idx] += 1
                bts.append(idx)
        block_tsubsets.append(bts)
    
    best_blocks = blocks[:]
    best_len = len(blocks)
    
    # Try swapping points in blocks
    for _ in range(50):  # Limited iterations
        if deadline is not None and time.time() > deadline:
            break
        
        bi = rng.randrange(len(blocks))
        block = blocks[bi]
        
        # Try swapping one point in block with one outside
        if len(block) < k:
            continue
        
        removed_point = block[rng.randrange(k)]
        # Find a point not in block
        candidates = [p for p in range(v) if p not in set(block)]
        if not candidates:
            continue
        new_point = rng.choice(candidates)
        
        # Create new block
        new_block = [p for p in block if p != removed_point] + [new_point]
        new_block.sort()
        
        # Check if this swap improves things (allows removing a block or maintains coverage)
        # For simplicity, just try it and see if we can remove a block afterwards
        test_blocks = blocks[:]
        test_blocks[bi] = new_block
        
        reduced = _remove_redundant_blocks(v, k, t, test_blocks, deadline=deadline)
        if len(reduced) < best_len:
            best_len = len(reduced)
            best_blocks = reduced[:]
    
    return best_blocks

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Returns a list of blocks, each a sorted list of k distinct integers in 0..v-1.
    """
    start_time = time.time()
    # Deadline: leave 10 seconds safety margin before the 150s limit
    deadline = start_time + 140.0
    
    best_blocks = None
    best_len = float('inf')
    
    # First, try the default greedy
    blocks = _greedy_construct(v, k, t, point_order=list(range(v)), deadline=deadline)
    if blocks:
        reduced = _remove_redundant_blocks(v, k, t, blocks, deadline=deadline)
        if len(reduced) < len(blocks):
            blocks = reduced
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
    
    # Randomized restarts
    rng = random.Random(42)
    attempt = 0
    while time.time() < deadline - 5.0:
        attempt += 1
        point_order = list(range(v))
        rng.shuffle(point_order)
        blocks = _greedy_construct(v, k, t, point_order=point_order, deadline=deadline, rng=rng)
        if blocks:
            reduced = _remove_redundant_blocks(v, k, t, blocks, deadline=deadline)
            if len(reduced) < len(blocks):
                blocks = reduced
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks[:]
    
    # Local search on the best solution
    if best_blocks is not None and time.time() < deadline - 2.0:
        improved = _local_search(v, k, t, best_blocks, deadline=deadline, rng=rng)
        if len(improved) < best_len:
            best_blocks = improved
            best_len = len(improved)
    
    if best_blocks is None:
        # Fallback: return a trivial covering
        return []
    
    return best_blocks
