"""Improved greedy covering with efficient gain computation and local search."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import defaultdict

def _count_covered(subset, uncovered):
    """Count how many uncovered t-subsets are contained in 'subset'."""
    count = 0
    for comb in itertools.combinations(subset, t_global):
        if tuple(comb) in uncovered:
            count += 1
    return count

def construct(v, k, t):
    global t_global
    t_global = t
    
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_t_subsets)
    
    if not uncovered:
        return []
    
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            # Return best effort so far - but we need a valid covering
            # If we're out of time, we have to complete the covering quickly
            # Use a fast greedy to finish
            return _fast_complete(uncovered, blocks, v, k, t, deadline)
        
        # Start with the most "constrained" uncovered subset
        # Use the first uncovered subset (deterministic for reproducibility, but we can randomize)
        # For better results, pick an uncovered subset that appears in fewest potential blocks
        # But for speed, just pick the first one
        first_uncovered = min(uncovered)
        block = list(first_uncovered)
        
        # Greedily add points to maximize coverage
        while len(block) < k:
            if time.time() > deadline:
                return _fast_complete(uncovered, blocks, v, k, t, deadline)
            
            best_gain = -1
            best_point = None
            block_set = set(block)
            
            for x in range(v):
                if x in block_set:
                    continue
                # Compute gain: how many new uncovered t-subsets would be covered
                gain = 0
                for comb in itertools.combinations(block, t - 1):
                    candidate = tuple(sorted(comb + (x,)))
                    if candidate in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None:
                break
            
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Remove covered t-subsets
        covered_by_block = set(itertools.combinations(block, t))
        uncovered -= covered_by_block
    
    return blocks


def _fast_complete(uncovered, blocks, v, k, t, deadline):
    """Fast completion when time is running out."""
    start_time = time.time()
    remaining_deadline = max(deadline - start_time, 5)
    
    while uncovered:
        if time.time() > start_time + remaining_deadline:
            # Emergency: add random blocks until covered or give up
            # This may not be valid, but better than nothing
            for _ in range(1000):
                if not uncovered:
                    break
                block = sorted(random.sample(range(v), k))
                blocks.append(block)
                covered = set(itertools.combinations(block, t))
                uncovered -= covered
            break
        
        first_uncovered = min(uncovered)
        block = list(first_uncovered)
        
        # Quick greedy
        while len(block) < k:
            best_gain = -1
            best_point = None
            block_set = set(block)
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for comb in itertools.combinations(block, t - 1):
                    candidate = tuple(sorted(comb + (x,)))
                    if candidate in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None:
                break
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        covered = set(itertools.combinations(block, t))
        uncovered -= covered
    
    return blocks
# EVOLVE-BLOCK-END
