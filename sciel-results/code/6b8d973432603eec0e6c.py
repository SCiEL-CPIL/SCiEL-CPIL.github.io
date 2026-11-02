import time
import random
import sys
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = 135.0  # safety margin
    
    # Precompute all t-subsets
    all_subsets = []
    for combo in _combinations_range(v, t):
        all_subsets.append(frozenset(combo))
    
    num_subsets = len(all_subsets)
    
    # Build index: for each subset, track if it's covered
    # We'll use a set of frozensets for uncovered
    uncovered = set(all_subsets)
    
    # Greedy initial solution
    blocks = []
    while uncovered:
        if time.time() > deadline:
            break
        # Pick the most constrained uncovered subset (one with fewest potential completions)
        # For speed, just pick the first one
        target = next(iter(uncovered))
        
        # Build a block containing target
        block = set(target)
        
        while len(block) < k:
            # Find point that covers most uncovered subsets
            best_point = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Count how many uncovered subsets would be covered by adding x
                new_block = block | {x}
                gain = 0
                for s in uncovered:
                    if s.issubset(new_block):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point >= 0:
                block.add(best_point)
        
        block = tuple(sorted(block))
        blocks.append(block)
        
        # Remove covered subsets
        block_set = frozenset(block)
        to_remove = []
        for s in uncovered:
            if s.issubset(block_set):
                to_remove.append(s)
        for s in to_remove:
            uncovered.discard(s)
    
    if time.time() > deadline:
        return blocks
    
    # Local search: try to replace blocks to reduce count while maintaining coverage
    # Strategy: try to find redundant blocks (blocks whose removal still leaves a valid cover)
    # and try to merge or replace blocks
    
    # First, let's try a simple approach: repeatedly try to remove blocks that are redundant
    def is_cover_valid(blks):
        if not blks:
            return num_subsets == 0
        covered = set()
        for b in blks:
            bset = frozenset(b)
            for s in all_subsets:
                if s.issubset(bset):
                    covered.add(s)
        return len(covered) == num_subsets
    
    # Try to find and remove redundant blocks
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            # Try removing block i
            candidate = blocks[:i] + blocks[i+1:]
            if is_cover_valid(candidate):
                blocks = candidate
                improved = True
                break
    
    # Now try to add a block that allows removing multiple others
    # This is more complex; for now, try a simple improvement:
    # For each pair of blocks, try to see if their union can be covered by one block
    # (only if their intersection plus some points forms a k-subset that covers all subsets covered by both)
    
    return blocks


def _combinations_range(n, r):
    """Generate combinations of range(n) taken r at a time."""
    if r > n or r < 0:
        return
    indices = list(range(r))
    yield tuple(indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(indices)
