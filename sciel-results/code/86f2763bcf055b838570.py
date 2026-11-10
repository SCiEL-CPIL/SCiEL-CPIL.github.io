import itertools
import time
import random
import numpy as np

DEADLINE_SECONDS = 115.0

def construct(v, k, t):
    start_time = time.time()
    
    if (v, k, t) != (19, 8, 5):
        return _generic_greedy(v, k, t, start_time)

    # Precomputation
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    n_t = len(all_t_subsets)
    idx_map = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Full mask for all t-subsets
    full_mask = (1 << n_t) - 1
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Precompute masks for each block
    # block_masks[i] is an int bitmask
    block_masks = [0] * n_blocks
    # blocks_containing[j] is a list of block indices containing t-subset j
    blocks_containing = [[] for _ in range(n_t)]
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(block, t):
            s_idx = idx_map[tuple(sub)]
            mask |= (1 << s_idx)
            blocks_containing[s_idx].append(b_idx)
        block_masks[b_idx] = mask

    best_blocks = []
    best_count = float('inf')
    
    # Multiple runs
    runs = 0
    while time.time() - start_time < DEADLINE_SECONDS - 5:
        runs += 1
        # Randomized Greedy
        uncovered = full_mask
        current_blocks = []
        
        # Shuffle candidates to introduce randomness? 
        # Or just pick random uncovered subset.
        
        while uncovered:
            # Pick a random set bit in uncovered
            # Efficient way: find a random index where bit is set
            # We can iterate through a shuffled list of indices? No, that's slow.
            # We can pick a random integer r, and find the r-th set bit? 
            # Simpler: pick a random subset index from the list of all subsets, 
            # check if it's in uncovered. If not, retry. 
            # Since density starts high and drops, this is okay initially.
            # Later, when few uncovered, this is inefficient.
            
            # Better: maintain a list of uncovered indices? 
            # Updating it is costly.
            
            # Alternative: Pick the first set bit? Deterministic.
            # To randomize, we can shuffle the order of `blocks_containing` lookups?
            # Or just pick a random 5-subset from the full list and check `uncovered >> idx & 1`.
            # If 0, try again.
            
            # Let's try picking a random index from 0 to n_t-1
            found_idx = -1
            attempts = 0
            while found_idx == -1 and attempts < 100:
                idx = random.randint(0, n_t - 1)
                if (uncovered >> idx) & 1:
                    found_idx = idx
                attempts += 1
            
            if found_idx == -1:
                # Fallback to linear scan if random fails (shouldn't happen often)
                for idx in range(n_t):
                    if (uncovered >> idx) & 1:
                        found_idx = idx
                        break
            
            if found_idx == -1:
                break # Should not happen if uncovered is non-zero
            
            # Get candidates
            candidates = blocks_containing[found_idx]
            
            best_gain = -1
            best_block_idx = -1
            # Randomize tie-breaking
            random.shuffle(candidates)
            
            for b_idx in candidates:
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx != -1:
                current_blocks.append(best_block_idx)
                uncovered &= ~block_masks[best_block_idx]
        
        # Pruning
        current_blocks = prune(current_blocks, block_masks, n_t)
        
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks[:]
            
        # If we hit a very good number, we can stop early?
        # Best known is 335. If we get < 335, it's a new record.
        # If we get 335, it's tied with best known.
        if best_count <= 335:
             break

    # Convert indices back to blocks
    result = [list(all_blocks[i]) for i in best_blocks]
    return result

def prune(blocks, block_masks, n_t):
    if not blocks:
        return []
    
    # Count coverage for each t-subset
    # We only care about subsets covered by at least one block in the current set
    # Actually, we need to know if removing a block leaves any subset uncovered.
    
    # Efficient Pruning:
    # 1. Compute coverage count for all subsets? 
    #    Only subsets in the union of current blocks matter.
    #    But we don't have the union explicitly, we have the list of blocks.
    
    # Let's use the "essential block" logic.
    # A block is essential if it is the ONLY block covering some subset.
    # We can iterate and remove non-essential blocks.
    
    current = blocks[:]
    changed = True
    while changed:
        changed = False
        # Compute for each subset in the current union, how many blocks cover it?
        # This is expensive if done from scratch each time.
        
        # Alternative: 
        # For each block B in current:
        #   Check if B is redundant.
        #   B is redundant if for every subset s in B, there exists another block B' in current\{B} such that s in B'.
        
        # Let's optimize:
        # Maintain a count array? 
        # Since n_t is 11628, we can use a numpy array of ints.
        
        # Re-implementation of prune with numpy for speed
        counts = np.zeros(n_t, dtype=np.int32)
        for b_idx in current:
            mask = block_masks[b_idx]
            # Set bits in counts? No, we need to increment.
            # Iterating bits of a large int is slow in Python.
            # Better: 
            # For each block, we know its subsets.
            # Let's precompute `block_subsets` as a list of lists of indices.
            
        # I didn't precompute `block_subsets` in the main loop, only masks.
        # I should precompute `block_subsets` as well.
        
    # Let's rewrite the precomputation to include subsets.
    
    return current
