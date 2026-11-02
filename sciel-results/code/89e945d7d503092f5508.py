"""Covering design C(v,k,t) with randomized greedy + local search."""
import itertools
import random
import time
import sys
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 125  # safety margin before 150s kill
    best_blocks = None
    best_count = float('inf')
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # Precompute for each block (as a frozenset) the list of t-subset indices it covers
    # We'll build this on-the-fly in greedy since blocks vary
    
    def get_block_tsub_indices(block):
        """Return list of t-subset indices covered by this block."""
        indices = []
        for ts in itertools.combinations(block, t):
            idx = tsub_index.get(ts)
            if idx is not None:
                indices.append(idx)
        return indices
    
    def greedy_build(seed, randomize_order=True):
        """Randomized greedy covering with possible randomization."""
        rng = random.Random(seed)
        uncovered_mask = np.ones(n_tsubs, dtype=np.int32)  # 1 = uncovered
        blocks = []
        
        # Randomize the order we consider t-subsets
        tsub_order = np.arange(n_tsubs)
        if randomize_order:
            rng.shuffle(tsub_order.tolist())
            tsub_order = np.array(tsub_order)
        
        while uncovered_mask.any():
            # Pick the first uncovered t-subset in our order
            first_uncovered = None
            for idx in tsub_order:
                if uncovered_mask[idx]:
                    first_uncovered = idx
                    break
            if first_uncovered is None:
                break
            
            block = list(all_tsubs[first_uncovered])
            
            # Add points greedily to maximize coverage
            while len(block) < k:
                in_block = set(block)
                candidates = [x for x in range(v) if x not in in_block]
                
                if not candidates:
                    break
                    
                if randomize_order:
                    rng.shuffle(candidates)
                
                best_gain = -1
                best_x = candidates[0]
                for x in candidates:
                    # Count how many uncovered t-subsets this point covers
                    gain = 0
                    for combo in itertools.combinations(block, t-1):
                        ts = tuple(sorted(combo + (x,)))
                        idx = tsub_index.get(ts)
                        if idx is not None and uncovered_mask[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain:
                        # Small random tiebreak
                        if rng.random() < 0.5:
                            best_x = x
                
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Remove covered t-subsets
            for idx in get_block_tsub_indices(block):
                uncovered_mask[idx] = 0
        
        return blocks
    
    # First, do a deterministic greedy to get a baseline
    blocks = greedy_build(0, randomize_order=False)
    best_blocks = blocks
    best_count = len(blocks)
    
    # Then try randomized restarts
    seed = 1
    while time.time() < deadline - 15:
        blocks = greedy_build(seed, randomize_order=True)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        seed += 1
    
    # Local search: try to remove blocks
    if best_blocks and time.time() < deadline - 10:
        best_blocks = local_search_optimize(best_blocks, v, k, t, all_tsubs, tsub_index, n_tsubs, deadline)
    
    return best_blocks if best_blocks else greedy_build(0, randomize_order=False)

def local_search_optimize(blocks, v, k, t, all_tsubs, tsub_index, n_tsubs, deadline):
    """Try to reduce the number of blocks via local search."""
    if not blocks:
        return blocks
    
    # Build coverage count for each t-subset using numpy
    cov_count = np.zeros(n_tsubs, dtype=np.int32)
    block_tsubs_list = []
    for block in blocks:
        indices = []
        for ts in itertools.combinations(block, t):
            idx = tsub_index.get(ts)
            if idx is not None:
                indices.append(idx)
        block_tsubs_list.append(indices)
        for idx in indices:
            cov_count[idx] += 1
    
    improved = True
    blocks_list = list(blocks)
    tsubs_list = list(block_tsubs_list)
    
    while improved and time.time() < deadline - 5:
        improved = False
        n_blocks = len(blocks_list)
        
        for i in range(n_blocks):
            if time.time() > deadline - 5:
                break
            
            # Try removing block i
            test_cov = cov_count.copy()
            for idx in tsubs_list[i]:
                test_cov[idx] -= 1
            
            # Check if all t-subsets still covered
            if test_cov.min() >= 1:
                # Remove block i
                blocks_list.pop(i)
                tsubs_list.pop(i)
                cov_count = test_cov
                improved = True
                break
    
    return blocks_list
