import itertools
import numpy as np
import time
import random

# Global cache for precomputed data
_precomputed_cache = {}

def _get_precomputed(v, k, t):
    key = (v, k, t)
    if key in _precomputed_cache:
        return _precomputed_cache[key]
    
    # 1. Generate all t-subsets and map to indices
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {tuple(s): i for i, s in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    
    # 2. Generate all k-subsets (blocks)
    k_subs = list(itertools.combinations(range(v), k))
    
    # 3. For each block, find indices of t-subsets it covers
    block_to_t_sets = []
    for b in k_subs:
        idxs = []
        for ts in itertools.combinations(b, t):
            idxs.append(t_sub_to_idx[tuple(ts)])
        block_to_t_sets.append(idxs)
        
    _precomputed_cache[key] = (k_subs, block_to_t_sets, num_t_subs)
    return _precomputed_cache[key]

def _greedy_cover(v, k, t, rng):
    k_subs, block_to_t_sets, num_t_subs = _get_precomputed(v, k, t)
    
    uncovered = np.ones(num_t_subs, dtype=bool)
    blocks = []
    
    while uncovered.any():
        best_block = -1
        best_gain = -1.0
        
        # Iterate through all possible blocks to find the one covering the most uncovered t-subsets
        for b_idx, t_idxs in enumerate(block_to_t_sets):
            gain = uncovered[t_idxs].sum()
            # Add small random noise to break ties and allow different restarts
            gain += rng.random() * 0.001
            
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
                
        if best_block == -1 or best_gain <= 0:
            break
            
        blocks.append(best_block)
        # Mark covered t-subsets
        uncovered[block_to_t_sets[best_block]] = False
        
    return blocks

def _local_search_reduce(blocks, v, k, t):
    k_subs, block_to_t_sets, num_t_subs = _get_precomputed(v, k, t)
    
    if not blocks:
        return blocks
        
    cover_count = np.zeros(num_t_subs, dtype=np.int32)
    for b_idx in blocks:
        cover_count[block_to_t_sets[b_idx]] += 1
        
    changed = True
    while changed:
        changed = False
        # Iterate in reverse to safely pop elements
        for i in range(len(blocks) - 1, -1, -1):
            b_idx = blocks[i]
            t_idxs = block_to_t_sets[b_idx]
            
            # If all t-subsets in this block are covered by at least one other block, we can remove it
            if np.min(cover_count[t_idxs]) > 1:
                cover_count[t_idxs] -= 1
                blocks.pop(i)
                changed = True
                
    return blocks

def construct(v, k, t):
    start_time = time.time()
    # Use 140s to be safe within 150s limit, leaving 10s margin
    deadline = start_time + 140.0
    
    # Ensure precomputed data is ready
    k_subs, block_to_t_sets, num_t_subs = _get_precomputed(v, k, t)
    
    best_blocks = None
    best_size = float('inf')
    
    restarts = 0
    while True:
        # Check time before starting a new expensive restart
        if time.time() > deadline:
            break
            
        rng = random.Random(restarts)
        
        # 1. Greedy Construction
        b_idx_list = _greedy_cover(v, k, t, rng)
        
        # 2. Local Search (Block Removal)
        b_idx_list = _local_search_reduce(b_idx_list, v, k, t)
        
        if len(b_idx_list) < best_size:
            best_size = len(b_idx_list)
            best_blocks = b_idx_list[:]
            
        restarts += 1
        
        # Optional: Early stop if we hit a very good score?
        # The known best is 151. If we find 151, we can try to beat it.
        # If we find 150, that's great.
        
    # Convert indices to actual blocks
    if best_blocks is None:
        # Fallback: return an empty list or a single block if something went wrong?
        # But greedy should always work.
        return []
        
    result = [list(k_subs[i]) for i in best_blocks]
    return result
