import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    # Allow 145 seconds for computation, leaving 5 seconds for cleanup/return
    deadline = start_time + 145.0
    
    # 1. Precomputation
    points = range(v)
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {tr: i for i, tr in enumerate(triples)}
    n_triples = len(triples)
    
    blocks = list(itertools.combinations(points, k))
    n_blocks = len(blocks)
    
    # Precompute block -> triple indices
    # all_block_triples: (n_blocks, C(k,t))
    # C(7,3) = 35
    t_combinations_in_block = list(itertools.combinations(range(k), t))
    # This is not quite right. The combinations of indices 0..k-1 map to points in the block.
    # We need to map the points in the block to the global triple indices.
    
    # Faster way:
    # For each block, generate its t-subsets, find their global indices.
    all_block_triples = np.zeros((n_blocks, int(np.math.comb(k, t))), dtype=np.int32)
    triple_block_indices = [[] for _ in range(n_triples)]
    
    for b_idx, block in enumerate(blocks):
        # Generate t-subsets of the block
        block_set = set(block)
        # The block is a tuple of points.
        # We need to find which global triples are subsets of this block.
        # Instead of iterating all global triples, iterate local t-combinations.
        # But we need the global index.
        # The points in the block are specific values.
        # We can just iterate the global triples? No, too slow.
        # We can iterate combinations of the block's points.
        local_triples = list(itertools.combinations(block, t))
        for i, lt in enumerate(local_triples):
            tr_idx = triple_to_idx[lt]
            all_block_triples[b_idx, i] = tr_idx
            triple_block_indices[tr_idx].append(b_idx)
            
    # Note: The order of triples in all_block_triples[b_idx] corresponds to 
    # the order of itertools.combinations(block, t).
    # This is consistent.
    
    best_blocks = None
    best_len = float('inf')
    
    # 2. Multi-start Greedy
    # Estimate time per start with vectorization
    # ~76 steps. Each step: 
    #   - np.array(cand_blocks)
    #   - fancy indexing all_block_triples
    #   - np.sum
    #   - np.argmax
    #   - update mask
    # This is very fast. Let's try 500 starts.
    
    n_starts = 500
    
    for i in range(n_starts):
        if time.time() > deadline - 10.0:
            break
            
        cover_count = np.zeros(n_triples, dtype=np.int32)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        
        current_blocks = []
        
        while np.any(uncovered_mask):
            # Pick a random uncovered triple
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            t_idx = uncovered_indices[np.random.choice(len(uncovered_indices))]
            
            # Get candidate blocks
            cand_blocks = triple_block_indices[t_idx]
            
            # Vectorized gain calculation
            cand_array = np.array(cand_blocks, dtype=np.int32)
            # Shape: (num_cands, 35)
            sub_triples = all_block_triples[cand_array]
            # Sum uncovered mask over the 35 triples for each candidate
            # uncovered_mask is (n_triples,)
            # sub_triples is (M, 35)
            # uncovered_mask[sub_triples] is (M, 35)
            gains = np.sum(uncovered_mask[sub_triples], axis=1)
            
            best_idx = np.argmax(gains)
            best_block = cand_blocks[best_idx]
            
            # Add best block
            current_blocks.append(best_block)
            
            # Update counts
            trs = all_block_triples[best_block]
            cover_count[trs] += 1
            # Update uncovered mask
            # A triple is uncovered if cover_count is 0
            uncovered_mask[trs] = (cover_count[trs] == 0)
            
        # Post-processing: Remove redundant blocks
        final_blocks = reduce_blocks(current_blocks, all_block_triples, n_triples)
        
        if len(final_blocks) < best_len:
            best_len = len(final_blocks)
            best_blocks = final_blocks
            
    # Convert block indices back to point lists
    if best_blocks is None:
        # Fallback: Return empty or a single block if no time was spent?
        # But we should have at least one.
        return []
        
    return [list(blocks[b]) for b in best_blocks]

def reduce_blocks(block_indices, all_block_triples, n_triples):
    # block_indices: list of block indices
    # Returns list of block indices, reduced
    
    current = list(block_indices)
    changed = True
    while changed:
        changed = False
        # Calculate cover count
        cover_count = np.zeros(n_triples, dtype=np.int32)
        for b in current:
            trs = all_block_triples[b]
            cover_count[trs] += 1
            
        # Identify redundant blocks
        # A block is redundant if it contains NO triple with cover_count == 1
        
        to_remove = []
        for b in current:
            trs = all_block_triples[b]
            # If any triple in this block has cover_count == 1, it's essential
            if np.any(cover_count[trs] == 1):
                continue
            else:
                to_remove.append(b)
                
        if to_remove:
            for b in to_remove:
                current.remove(b)
            changed = True
            
    return current
