"""Covering design C(24,7,3) via greedy with efficient gain computation and local search."""
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Precompute all t-subsets as frozensets for fast lookup
    all_tsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    uncovered = set(all_tsets)
    blocks = []
    
    # Precompute: for each point x and each (t-1)-subset R, the t-set {x} ∪ R
    # Map from (x, R) to frozenset({x} ∪ R) for quick lookup
    # But that's v * C(v, t-1) = 24 * C(24,2) = 24*276 = 6624 entries, manageable
    
    # Actually, let's precompute a dict: for each t-set, which points are in it
    # And for gain: given partial block B and candidate x, 
    # gain = count of (t-1)-subsets R of B such that {x}∪R is in uncovered
    
    # Precompute all (t-1)-subsets as sorted tuples for fast iteration
    t_minus_1_subsets = list(itertools.combinations(range(v), t - 1))
    
    while uncovered and time.time() < deadline - 30:  # reserve 30s for refinement
        # Pick the first uncovered t-set to seed a new block
        seed_tset = next(iter(uncovered))
        block = list(seed_tset)
        
        while len(block) < k and time.time() < deadline - 30:
            block_set = set(block)
            best_x = -1
            best_gain = -1
            
            # For each candidate x not in block
            for x in range(v):
                if x in block_set:
                    continue
                # gain = number of (t-1)-subsets R of block such that {x}∪R is uncovered
                gain = 0
                for R in itertools.combinations(block, t - 1):
                    ts = frozenset(list(R) + [x])
                    if ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1 or best_gain == 0:
                # No point adds coverage; pick randomly
                remaining = [x for x in range(v) if x not in block_set]
                if remaining:
                    best_x = random.choice(remaining)
                else:
                    break
            block.append(best_x)
        
        block.sort()
        
        # Remove all covered t-sets
        covered = set()
        for c in itertools.combinations(block, t):
            covered.add(frozenset(c))
        uncovered -= covered
        blocks.append(block)
    
    # Ensure full coverage
    if uncovered:
        for ts in list(uncovered):
            if time.time() >= deadline - 5:
                break
            block = list(ts)
            while len(block) < k:
                x = random.randint(0, v - 1)
                if x not in block:
                    block.append(x)
            block.sort()
            if block not in blocks:
                blocks.append(block)
                for c in itertools.combinations(block, t):
                    uncovered.discard(frozenset(c))
    
    # Local search refinement: try to reduce block count by finding redundant blocks
    # or improving individual blocks
    if time.time() < deadline - 5:
        blocks = _local_search(blocks, v, k, t, deadline)
    
    return blocks

def _local_search(blocks, v, k, t, deadline):
    """Try to improve the covering by local search: 
    1. Remove redundant blocks (if a block's t-sets are all covered by other blocks)
    2. Try swapping points in blocks to improve coverage efficiency
    """
    blocks = [list(b) for b in blocks]
    n = len(blocks)
    
    # Precompute: for each block, the set of t-sets it covers
    block_tsets = []
    for block in blocks:
        tsets = set()
        for c in itertools.combinations(block, t):
            tsets.add(frozenset(c))
        block_tsets.append(tsets)
    
    # Try removing redundant blocks
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(len(blocks)):
            if time.time() >= deadline - 5:
                break
            # Check if block i is redundant: all its t-sets are covered by other blocks
            other_covered = set()
            for j in range(len(blocks)):
                if j == i:
                    continue
                other_covered.update(block_tsets[j])
            
            if block_tsets[i].issubset(other_covered):
                # Block i is redundant
                blocks.pop(i)
                block_tsets.pop(i)
                improved = True
                break  # restart from beginning since indices changed
    
    # Try improving individual blocks by swapping points
    # For each block, try replacing one point with another to see if we can
    # maintain coverage while potentially enabling removal of another block
    if time.time() < deadline - 10:
        for iteration in range(3):  # a few passes
            if time.time() >= deadline - 5:
                break
            progress = False
            for i in range(len(blocks)):
                if time.time() >= deadline - 5:
                    break
                block = blocks[i]
                block_set = set(block)
                
                # Try each point in the block
                for pos in range(k):
                    if time.time() >= deadline - 5:
                        break
                    old_point = block[pos]
                    
                    # Try replacing with each other point
                    for new_point in range(v):
                        if new_point in block_set:
                            continue
                        
                        # Create new block
                        new_block = list(block)
                        new_block[pos] = new_point
                        new_block.sort()
                        
                        # Check if this new block still covers all t-sets that were 
                        # only covered by the old block
                        new_tsets = set()
                        for c in itertools.combinations(new_block, t):
                            new_tsets.add(frozenset(c))
                        
                        # Find t-sets covered by old block but not new block
                        lost_tsets = block_tsets[i] - new_tsets
                        
                        # Check if all lost t-sets are covered by other blocks
                        redundant = True
                        if lost_tsets:
                            other_covered = set()
                            for j in range(len(blocks)):
                                if j == i:
                                    continue
                                other_covered.update(block_tsets[j])
                            if not lost_tsets.issubset(other_covered):
                                redundant = False
                        
                        if redundant:
                            # Accept the swap
                            blocks[i] = new_block
                            block_tsets[i] = new_tsets
                            progress = True
                            break  # restart inner loop
                    
                    if progress:
                        break
                
                if progress:
                    break
            
            if not progress:
                break
    
    return blocks
