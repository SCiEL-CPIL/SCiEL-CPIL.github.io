import itertools
import numpy as np
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for safety/return
    
    # 1. Precompute all t-subsets (triples) and map them to indices
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # 2. Precompute all k-subsets (blocks) and map them to their covered triple indices
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Store the indices of triples covered by each block
    # Shape: (num_blocks, C(k, t))
    # For v=24, k=7, t=3, C(7,3)=35.
    block_triple_indices = np.empty((num_blocks, 35), dtype=np.int32)
    
    # Precomputation loop
    for i, block in enumerate(all_blocks):
        # Generate all t-subsets of this block
        # Since k and t are small, itertools is fine
        indices = []
        for sub in itertools.combinations(block, t):
            indices.append(triple_to_idx[sub])
        block_triple_indices[i] = indices
        
        # Check time during precomputation just in case
        if time.time() > deadline:
            # Fallback to a simple greedy if precomputation is too slow (unlikely)
            # But for v=24, it should be fast.
            pass

    # 3. Greedy Set Cover
    uncovered = np.ones(num_triples, dtype=bool)
    selected_blocks = []
    
    while uncovered.any():
        if time.time() > deadline:
            break
            
        # Calculate gain for each block: number of currently uncovered triples it covers
        # uncovered[block_triple_indices] creates a boolean array of shape (num_blocks, 35)
        # .sum(axis=1) sums along the triple dimension
        gains = uncovered[block_triple_indices].sum(axis=1)
        
        # Find the block with the maximum gain
        best_idx = np.argmax(gains)
        
        # If no block covers any new triple, we have a problem (shouldn't happen)
        if gains[best_idx] == 0:
            break
            
        # Add the block to our solution
        selected_blocks.append(all_blocks[best_idx])
        
        # Mark the triples covered by this block as covered
        covered_indices = block_triple_indices[best_idx]
        uncovered[covered_indices] = False
        
    return selected_blocks
