import itertools
import numpy as np
from math import comb

def construct(v, k, t):
    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)
    
    # Store block indices for fast access
    # block_triple_indices[i, j] is the index of the j-th triple in block i
    block_triple_indices = np.empty((n_blocks, k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            block_triple_indices[i, j] = triple_to_idx[sub]
            
    # Precompute scarcity: how many blocks cover each triple
    # This helps in tie-breaking: prefer blocks covering "rare" triples
    triple_scarcity = np.zeros(n_triples, dtype=np.int32)
    # We can compute this by counting occurrences in block_triple_indices
    # But block_triple_indices is large. 
    # Alternative: just count during the greedy loop? No, that's slow.
    # Let's just use random tie-breaking for now, it's robust.
    # Or, we can compute scarcity once.
    # The matrix is (n_blocks, k_t). We can sum over rows? No, we need counts per column value.
    # It's easier to just use random tie-breaking which is proven to work well with restarts.
    
    def greedy_cover(random_seed):
        rng = np.random.RandomState(random_seed)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        remaining = n_triples
        
        while remaining > 0:
            # Calculate gains
            gains = np.sum(uncovered_mask[block_triple_indices], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly pick one candidate
            best_idx = candidates[rng.randint(len(candidates))]
            
            # Update state
            block_idx = block_triple_indices[best_idx]
            uncovered_mask[block_idx] = False
            remaining -= max_gain
            selected_blocks.append(best_idx) # Store index for efficiency
            
        # Post-processing: Remove redundant blocks
        # A block is redundant if all its triples are covered by other selected blocks
        # We need to check coverage counts.
        if selected_blocks:
            # Create a coverage count array for the selected blocks
            # We can do this by iterating and adding, or using sparse matrices if available, 
            # but standard numpy is safer.
            # n_triples is ~2000, len(selected) is ~70-80. 
            # We can just loop through selected blocks and count.
            
            cover_count = np.zeros(n_triples, dtype=np.int32)
            for b_idx in selected_blocks:
                cover_count[block_triple_indices[b_idx]] += 1
            
            # Now, iterate through selected blocks and see if any triple has count == 1
            # If a block has NO triple with count == 1, it is redundant.
            # We must be careful: removing a block changes the counts for others.
            # Simple approach: 
            # 1. Identify all blocks that are potentially redundant (no unique triple).
            # 2. Remove them one by one, updating counts.
            
            # Let's do a simple pass:
            # For each block in the list, check if it has a unique triple.
            # If not, mark for removal.
            # Then remove all marked blocks.
            # Then repeat? One pass is usually enough for greedy outputs.
            
            to_remove = []
            for b_idx in selected_blocks:
                # Check if any triple in this block has cover_count == 1
                # If yes, this block is essential.
                # If no, it's redundant.
                block_triples = block_triple_indices[b_idx]
                # Check if any count is 1
                if not np.any(cover_count[block_triples] == 1):
                    to_remove.append(b_idx)
                    
            if to_remove:
                # Remove them
                selected_blocks = [b for b in selected_blocks if b not in to_remove]
                
        return [list(all_blocks[b_idx]) for b_idx in selected_blocks]

    # Run multiple times
    best_blocks = None
    best_len = float('inf')
    
    n_restarts = 20
    
    for i in range(n_restarts):
        blocks = greedy_cover(i)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    return best_blocks
