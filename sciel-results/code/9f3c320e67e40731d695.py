import itertools
import random
import time
import sys

_CACHE = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    # Generate all blocks
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Map each triple to a unique bit index
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {tri: i for i, tri in enumerate(triples)}
    n_triples = len(triples)
    
    # Precompute masks for each block
    # A block mask is an integer where bit j is 1 if the block contains the j-th triple
    block_masks = [0] * n_blocks
    for i, block in enumerate(all_blocks):
        mask = 0
        for tri in itertools.combinations(block, t):
            idx = triple_to_idx[tri]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    full_mask = (1 << n_triples) - 1
    
    _CACHE[key] = (all_blocks, block_masks, full_mask)
    return _CACHE[key]

def _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, seed, deadline):
    rng = random.Random(seed)
    uncovered = full_mask
    blocks_used = []
    n_blocks = len(all_blocks)
    
    # Pre-compute list of block indices for shuffling
    indices = list(range(n_blocks))
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Shuffle the order of block evaluation to diversify the greedy path
        rng.shuffle(indices)
        
        best_block_idx = -1
        max_gain = 0
        
        for idx in indices:
            gain = (block_masks[idx] & uncovered).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_block_idx = idx
                # Early exit if we found a block that covers all remaining? 
                # Unlikely for large uncovered, but if max_gain == uncovered.bit_count(), we can stop searching for better
                if max_gain == uncovered.bit_count():
                    break
            
        if best_block_idx == -1:
            # Should not happen if covering is possible
            break
            
        blocks_used.append(best_block_idx)
        uncovered &= ~block_masks[best_block_idx]
        
    return blocks_used

def _reduce_blocks(v, k, t, block_indices, block_masks, full_mask):
    if not block_indices:
        return []
        
    current_blocks = list(block_indices)
    changed = True
    
    while changed:
        changed = False
        new_current = []
        n = len(current_blocks)
        
        for i, idx in enumerate(current_blocks):
            # Check if block i is redundant
            # Union of all blocks except i
            union_others = 0
            for j, other_idx in enumerate(current_blocks):
                if i != j:
                    union_others |= block_masks[other_idx]
            
            if (union_others & full_mask) == full_mask:
                # Redundant
                changed = True
            else:
                new_current.append(idx)
                
        current_blocks = new_current
        
    return current_blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115 # Safety margin (150s limit, 35s buffer)
    
    # Precompute
    all_blocks, block_masks, full_mask = _precompute(v, k, t)
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    # Run greedy multiple times
    while time.time() < deadline - 5: # Leave 5s for final reduction and safety
        blocks = _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, seed, deadline - 5)
        seed += 1
        
        if not blocks:
            continue
            
        # Reduce
        reduced_blocks = _reduce_blocks(v, k, t, blocks, block_masks, full_mask)
        
        if len(reduced_blocks) < best_len:
            best_len = len(reduced_blocks)
            best_blocks = reduced_blocks
            
    # If we have a result, convert indices to actual block lists
    if best_blocks is None:
        # Fallback: return a simple valid covering if we failed (shouldn't happen)
        # Do one fast greedy without shuffle overhead if time is tight? 
        # Just run one more greedy
        blocks = _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, 999, time.time() + 5)
        best_blocks = blocks
        
    # Format output: list of lists of ints
    result = [list(all_blocks[idx]) for idx in best_blocks]
    
    return result
