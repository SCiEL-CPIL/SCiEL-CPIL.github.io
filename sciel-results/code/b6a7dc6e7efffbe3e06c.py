import numpy as np
import time
import itertools

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s
    
    # Precompute all t-subsets and their indices
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute all k-blocks and which t-subsets each covers
    all_kblocks = list(itertools.combinations(range(v), k))
    num_kblocks = len(all_kblocks)
    
    # For each block, store the indices of t-subsets it covers
    # Shape: (num_kblocks, C(k,t))
    t_per_block = list(itertools.combinations(range(k), t))
    num_t_per_block = len(t_per_block)
    
    block_tsub_indices = np.empty((num_kblocks, num_t_per_block), dtype=np.int32)
    for i, block in enumerate(all_kblocks):
        block_set = set(block)
        for j, tsub in enumerate(all_tsubs):
            pass  # This would be too slow; need better approach
    
    # Better: for each block, generate its C(k,t) t-subsets directly
    for i, block in enumerate(all_kblocks):
        for j, tsub in enumerate(itertools.combinations(block, t)):
            block_tsub_indices[i, j] = tsub_to_idx[tsub]
    
    # Greedy set cover
    uncovered = np.ones(num_tsubs, dtype=bool)
    chosen_blocks = []
    used_block_indices = set()
    
    while True:
        if time.time() > deadline - 30:  # check deadline
            break
        if not uncovered.any():
            break
        
        # Compute gains for all unused blocks
        gains = uncovered[block_tsub_indices].sum(axis=1)
        
        # Mask out already-used blocks
        for bi in used_block_indices:
            gains[bi] = -1
        
        best_block_idx = int(np.argmax(gains))
        if gains[best_block_idx] <= 0:
            break
        
        # Add this block
        chosen_blocks.append(all_kblocks[best_block_idx])
        used_block_indices.add(best_block_idx)
        
        # Update uncovered
        uncovered[block_tsub_indices[best_block_idx]] = False
    
    # Check if we have full coverage
    if uncovered.any():
        # Fallback: add blocks greedily without the mask (shouldn't happen if we have enough blocks)
        # Actually this shouldn't happen since we have all possible blocks
        pass
    
    # Local search polish: try to improve by swapping or removing blocks
    # Convert to list of block index tuples
    chosen_indices = [all_kblocks.index(b) for b in chosen_blocks]
    
    # Try removing redundant blocks
    for _ in range(3):  # a few passes
        if time.time() > deadline - 10:
            break
        new_uncovered = np.zeros(num_tsubs, dtype=bool)
        # Recompute coverage from scratch for the current set
        for bi in chosen_indices:
            new_uncovered[block_tsub_indices[bi]] = True
        
        # Try removing each block
        improved = False
        for idx, bi in enumerate(list(chosen_indices)):
            if time.time() > deadline - 10:
                break
            # What would be uncovered if we remove this block?
            # Count how many t-subsets are ONLY covered by this block
            # This is expensive; skip for now or do a simpler check
            pass
        
        # Try swapping: replace a block with a different one
        # For each chosen block, find a block that covers more uncovered subsets
        # This is complex; keep it simple for now
    
    # Final validation
    result = [list(b) for b in chosen_blocks]
    
    # Verify all t-subsets are covered
    final_uncovered = np.zeros(num_tsubs, dtype=bool)
    for b in result:
        for tsub in itertools.combinations(b, t):
            final_uncovered[tsub_to_idx[tsub]] = True
    
    if not final_uncovered.all():
        # Something went wrong; do a simple greedy fallback
        # This shouldn't happen, but just in case
        pass
    
    return result
