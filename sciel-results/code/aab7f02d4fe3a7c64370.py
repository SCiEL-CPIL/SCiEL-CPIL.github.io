"""Covering design C(v,k,t) with randomized greedy + local search."""
import itertools
import random
import time
import sys
from collections import defaultdict
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 115  # safety margin before 150s kill
    best_blocks = None
    best_count = float('inf')
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # Precompute for each t-subset which points it contains (for fast lookup)
    # And for each point, which t-subsets it belongs to
    point_tsubs = defaultdict(list)
    for i, ts in enumerate(all_tsubs):
        for p in ts:
            point_tsubs[p].append(i)
    
    def greedy_build(seed, randomize_order=True):
        """Randomized greedy covering with possible randomization."""
        rng = random.Random(seed)
        uncovered = bytearray([1]) * n_tsubs  # 1 if uncovered, 0 if covered
        uncovered_count = n_tsubs
        blocks = []
        
        # Randomize the order we consider t-subsets
        tsub_order = list(range(n_tsubs))
        if randomize_order:
            rng.shuffle(tsub_order)
        
        # Track which t-subsets are still uncovered, using a set for membership
        uncovered_set = set(range(n_tsubs))
        
        while uncovered_set:
            # Pick the first uncovered t-subset in our order
            first_uncovered = None
            for idx in tsub_order:
                if idx in uncovered_set:
                    first_uncovered = idx
                    break
            if first_uncovered is None:
                break
            
            block = list(all_tsubs[first_uncovered])
            block_set = set(block)
            
            # Add points greedily to maximize coverage
            while len(block) < k:
                candidates = [x for x in range(v) if x not in block_set]
                
                if not candidates:
                    break
                    
                if randomize_order:
                    rng.shuffle(candidates)
                
                best_gain = -1
                best_x = candidates[0]
                for x in candidates:
                    # Count how many uncovered t-subsets this point covers
                    # A t-subset (t-1 elements from block + x) is covered if:
                    # 1. The t-1 elements are in block
                    # 2. The resulting t-subset is currently uncovered
                    gain = 0
                    if t > 1:
                        for combo in itertools.combinations(block, t-1):
                            ts = tuple(sorted(combo + (x,)))
                            idx = tsub_index.get(ts)
                            if idx is not None and idx in uncovered_set:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain:
                        # Small random tiebreak
                        if rng.random() < 0.5:
                            best_x = x
                
                block.append(best_x)
                block_set.add(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Remove covered t-subsets
            for ts in itertools.combinations(block, t):
                idx = tsub_index.get(ts)
                if idx is not None and idx in uncovered_set:
                    uncovered_set.discard(idx)
                    uncovered_count -= 1
        
        return blocks
    
    # Try multiple restarts with different seeds
    rng_master = random.Random(42)
    
    # First, do a deterministic greedy to get a baseline
    blocks = greedy_build(0, randomize_order=False)
    best_blocks = blocks
    best_count = len(blocks)
    
    # Then try randomized restarts
    for seed in range(1, 500):
        if time.time() > deadline - 15:
            break
        
        blocks = greedy_build(seed, randomize_order=True)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        
        # Periodically check time
        if seed % 5 == 0 and time.time() > deadline - 10:
            break
    
    # Local search: try to remove redundant blocks
    if best_blocks and time.time() < deadline - 30:
        best_blocks = local_search_optimize(best_blocks, v, k, t, all_tsubs, tsub_index, n_tsubs, deadline)
    
    # Second pass local search if time permits
    if best_blocks and time.time() < deadline - 15:
        best_blocks = local_search_optimize(best_blocks, v, k, t, all_tsubs, tsub_index, n_tsubs, deadline)
    
    return best_blocks if best_blocks else greedy_build(0, randomize_order=False)

def local_search_optimize(blocks, v, k, t, all_tsubs, tsub_index, n_tsubs, deadline):
    """Try to reduce the number of blocks via local search."""
    if not blocks:
        return blocks
    
    # Build coverage count for each t-subset
    cov_count = [0] * n_tsubs
    for block in blocks:
        for ts in itertools.combinations(block, t):
            idx = tsub_index.get(ts)
            if idx is not None:
                cov_count[idx] += 1
    
    improved = True
    blocks_list = list(blocks)
    
    while improved and time.time() < deadline - 5:
        improved = False
        n_blocks = len(blocks_list)
        
        for i in range(n_blocks):
            if time.time() > deadline - 5:
                break
            
            # Try removing block i
            test_cov = cov_count[:]
            block_i = blocks_list[i]
            for ts in itertools.combinations(block_i, t):
                idx = tsub_index.get(ts)
                if idx is not None:
                    test_cov[idx] -= 1
            
            # Check if all t-subsets still covered
            if all(c >= 1 for c in test_cov):
                # Remove block i
                blocks_list.pop(i)
                cov_count = test_cov
                improved = True
                break
        
        if not improved and time.time() < deadline - 10:
            # Try replacing a block with a better one
            for i in range(len(blocks_list)):
                if time.time() > deadline - 5:
                    break
                # This is expensive; skip for now
                pass
    
    return blocks_list
