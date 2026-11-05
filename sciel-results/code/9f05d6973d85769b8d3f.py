import itertools
import numpy as np
import time
from math import comb

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin
    
    # 1. Map triples to indices
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    # 2. Generate all blocks and their triple indices
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)
    
    # Build block_triple_indices as a flat array with offsets for efficient access
    # Also build an inverse: for each triple, which blocks contain it
    # For memory efficiency, use int32
    block_triple_indices = np.empty((n_blocks, k_t), dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_triple_indices[i, j] = triple_to_idx[sub]
    
    # 3. Greedy selection with efficient gain computation
    # Maintain: uncovered_count per triple (how many selected blocks cover it, 0 = uncovered)
    # Actually, we need: for each triple, is it covered? And for each block, how many uncovered triples does it contain?
    
    # More efficient approach: 
    # - covered_mask: bool array, True if triple is covered
    # - For gains: sum over each block of (not covered_mask[triple]) for triples in block
    # We can precompute for each block the list of triple indices
    # Then gains = sum of (1 - covered_mask) over those triples
    
    # Even better: maintain a "uncovered_set" and for each block compute intersection size
    # Using numpy: gains = np.sum(~covered_mask[block_triple_indices], axis=1)
    # But this is still O(n_blocks * k_t) per iteration
    
    # Alternative: maintain for each triple a count of how many *selected* blocks cover it
    # When we select a block, we increment counts for all its triples
    # A triple is "uncovered" if count == 0
    # Gain of a block = number of triples in it with count == 0
    
    # Let's use: coverage_count[i] = number of selected blocks covering triple i
    # uncovered = coverage_count == 0
    # gain for block b = sum(uncovered[triple_indices[b]])
    
    coverage_count = np.zeros(n_triples, dtype=np.int32)
    selected_block_indices = []
    
    remaining = n_triples
    
    # Precompute for faster access: we'll use the block_triple_indices directly
    while remaining > 0:
        if time.time() > deadline:
            break
        
        # Compute gains: for each block, count how many of its triples are still uncovered
        # uncovered = (coverage_count == 0)
        # gains[b] = sum((coverage_count == 0)[block_triple_indices[b]])
        # This is: sum(1 - (coverage_count > 0)) over triples in block b
        
        # Vectorized: 
        # is_uncovered = coverage_count == 0  # bool array of length n_triples
        # gains = np.sum(is_uncovered[block_triple_indices], axis=1)
        
        is_uncovered = (coverage_count == 0)
        gains = np.sum(is_uncovered[block_triple_indices], axis=1)
        
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        best_idx = int(np.argmax(gains))
        
        # Update coverage counts for triples in the selected block
        block_triples = block_triple_indices[best_idx]
        coverage_count[block_triples] += 1
        
        # Update remaining: number of triples that went from uncovered to covered
        # These are the triples in block_triples that were previously uncovered (count was 0)
        newly_covered = np.sum(coverage_count[block_triples] == 1)
        remaining -= newly_covered
        
        selected_block_indices.append(best_idx)
        
    # Convert to actual blocks
    selected_blocks = [list(all_blocks[i]) for i in selected_block_indices]
    
    # 4. Local improvement: try to remove or swap blocks
    # Simple pass: check if any selected block is redundant (all its triples covered by other blocks)
    # If so, remove it
    if len(selected_blocks) > 0 and time.time() < deadline - 5.0:
        # Rebuild coverage from selected blocks
        n_sel = len(selected_blocks)
        # coverage_count2[i] = number of selected blocks covering triple i
        coverage_count2 = np.zeros(n_triples, dtype=np.int32)
        for bi in selected_block_indices:
            coverage_count2[block_triple_indices[bi]] += 1
        
        # Try removing each block
        for idx in range(n_sel):
            if time.time() > deadline - 5.0:
                break
            # Check if removing block idx leaves all triples covered
            # A triple is covered by block idx if coverage_count2[triple] >= 1 and this block contributes
            # After removal, triple is uncovered if coverage_count2[triple] == 1 and this block was the only one
            # So: check if any triple has coverage_count2[triple] == 1 AND is in block idx
            block_triples = block_triple_indices[selected_block_indices[idx]]
            # Triples that would become uncovered: those with count==1 in this block
            would_become_uncovered = np.sum(coverage_count2[block_triples] == 1)
            if would_become_uncovered == 0:
                # Block is redundant, remove it
                selected_block_indices.pop(idx)
                # Rebuild coverage
                coverage_count2 = np.zeros(n_triples, dtype=np.int32)
                for bi in selected_block_indices:
                    coverage_count2[block_triple_indices[bi]] += 1
                # Restart the removal pass since indices changed
                break
        
        # Another pass after removal
        if len(selected_blocks) > 0 and time.time() < deadline - 3.0:
            n_sel = len(selected_block_indices)
            coverage_count2 = np.zeros(n_triples, dtype=np.int32)
            for bi in selected_block_indices:
                coverage_count2[block_triple_indices[bi]] += 1
            
            for idx in range(n_sel):
                if time.time() > deadline - 3.0:
                    break
                block_triples = block_triple_indices[selected_block_indices[idx]]
                would_become_uncovered = np.sum(coverage_count2[block_triples] == 1)
                if would_become_uncovered == 0:
                    selected_block_indices.pop(idx)
                    coverage_count2 = np.zeros(n_triples, dtype=np.int32)
                    for bi in selected_block_indices:
                        coverage_count2[block_triple_indices[bi]] += 1
                    break
    
    selected_blocks = [list(all_blocks[i]) for i in selected_block_indices]
    
    return selected_blocks
