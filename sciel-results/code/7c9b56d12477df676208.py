"""Improved covering design constructor for C(24,7,3) with local search."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np
from collections import defaultdict

def _all_t_subsets(v, t):
    """Generate all t-subsets of {0,...,v-1} as tuples."""
    return list(itertools.combinations(range(v), t))

def _block_t_subsets(block, t):
    """Generate all t-subsets contained in a block."""
    return [tuple(sorted(c)) for c in itertools.combinations(block, t)]

def _count_covered(blocks, t):
    """Count how many t-subsets are covered by the given blocks."""
    covered = set()
    for block in blocks:
        for c in itertools.combinations(block, t):
            covered.add(tuple(sorted(c)))
    return len(covered)

def _uncovered_count(blocks, all_subsets, t):
    """Return the number of uncovered t-subsets."""
    covered = set()
    for block in blocks:
        for c in itertools.combinations(block, t):
            covered.add(tuple(sorted(c)))
    return len(all_subsets) - len(covered)

def _greedy_construct(v, k, t, all_subsets, deadline):
    """Greedy covering: open blocks to cover uncovered t-subsets."""
    uncovered = set(all_subsets)
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
        
        # Pick the least-covered t-subset to start a new block
        # For efficiency, just pick the first one
        start_subset = min(uncovered)
        block = list(start_subset)
        
        # Greedily add points to maximize coverage of uncovered subsets
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Count how many uncovered t-subsets would be covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    candidate = tuple(sorted(rest + (x,)))
                    if candidate in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1 or best_gain <= 0:
                # No useful point to add, just fill with arbitrary points
                for x in range(v):
                    if x not in block:
                        block.append(x)
                        if len(block) == k:
                            break
                break
            
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Remove covered subsets from uncovered
        for c in itertools.combinations(block, t):
            uncovered.discard(tuple(sorted(c)))
    
    return blocks

def _local_search(blocks, v, k, t, all_subsets, deadline, max_iter=500):
    """Local search: try to remove redundant blocks and swap points."""
    blocks = [list(b) for b in blocks]
    n = len(blocks)
    
    # Precompute coverage counts
    def build_coverage(blocks):
        cov = defaultdict(int)
        for block in blocks:
            for c in itertools.combinations(block, t):
                cov[tuple(sorted(c))] += 1
        return cov
    
    cov = build_coverage(blocks)
    uncovered_count = sum(1 for s in all_subsets if cov.get(s, 0) == 0)
    
    if uncovered_count > 0:
        # Need to fill gaps first - add greedy blocks
        uncovered = set(s for s in all_subsets if cov.get(s, 0) == 0)
        while uncovered and time.time() < deadline:
            start_subset = min(uncovered)
            block = list(start_subset)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1 or best_gain <= 0:
                    for x in range(v):
                        if x not in block:
                            block.append(x)
                            if len(block) == k:
                                break
                    break
                block.append(best_x)
            block.sort()
            blocks.append(block)
            for c in itertools.combinations(block, t):
                uncovered.discard(tuple(sorted(c)))
            cov = build_coverage(blocks)
            uncovered_count = sum(1 for s in all_subsets if cov.get(s, 0) == 0)
        
        if uncovered_count > 0:
            return blocks  # Still not covered, return as-is
    
    # Now try to remove redundant blocks
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            # Try removing block i
            new_blocks = blocks[:i] + blocks[i+1:]
            new_cov = build_coverage(new_blocks)
            new_uncovered = sum(1 for s in all_subsets if new_cov.get(s, 0) == 0)
            if new_uncovered == 0:
                blocks = new_blocks
                cov = new_cov
                improved = True
    
    # Try point swaps to enable further removals
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        # Try swapping a point in a block with another point
        best_swap = None
        best_new_count = len(blocks)
        
        for i, block in enumerate(blocks):
            for j, x in enumerate(block):
                for y in range(v):
                    if y in block:
                        continue
                    new_block = block[:]
                    new_block[j] = y
                    new_block.sort()
                    new_blocks = blocks[:i] + [new_block] + blocks[i+1:]
                    new_cov = build_coverage(new_blocks)
                    new_uncovered = sum(1 for s in all_subsets if new_cov.get(s, 0) == 0)
                    if new_uncovered == 0 and len(new_blocks) < best_new_count:
                        best_new_count = len(new_blocks)
                        best_swap = (i, j, y)
        
        if best_swap is None:
            break
        
        i, j, y = best_swap
        block = blocks[i][:]
        block[j] = y
        block.sort()
        blocks[i] = block
    
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    all_subsets = _all_t_subsets(v, t)
    
    # Phase 1: Greedy construction
    blocks = _greedy_construct(v, k, t, all_subsets, deadline)
    
    if time.time() > deadline - 20:
        return blocks
    
    # Phase 2: Local search for improvement
    blocks = _local_search(blocks, v, k, t, all_subsets, deadline, max_iter=300)
    
    return blocks
# EVOLVE-BLOCK-END
