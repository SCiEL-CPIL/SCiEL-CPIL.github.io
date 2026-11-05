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
    block_triple_indices = np.empty((n_blocks, k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            block_triple_indices[i, j] = triple_to_idx[sub]
            
    def greedy_cover(random_seed, top_k=10):
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
                
            # Find top K candidates
            # argsort is expensive for large n_blocks, but n_blocks for C(24,7) is 346k.
            # argsort of 346k floats is ~10ms. 76 iterations * 20 restarts = 1520 sorts.
            # 1520 * 10ms = 15 seconds. This might be tight for 120s limit if precomputation is slow.
            # Let's optimize: only look at top K.
            
            # Alternative: use np.argpartition for top K
            if n_blocks > top_k:
                # Get indices of top K gains
                top_indices = np.argpartition(gains, -top_k)[-top_k:]
            else:
                top_indices = np.arange(n_blocks)
                
            # Get the actual gains for these top candidates
            top_gains = gains[top_indices]
            
            # Filter out any with 0 gain (safety, though max_gain>0 ensures at least one is >0)
            # We want to pick from candidates with gain close to max_gain?
            # Or just uniformly from top K?
            # Let's pick uniformly from the top K to encourage diversity.
            
            best_idx_local = rng.randint(len(top_indices))
            best_idx = top_indices[best_idx_local]
            
            # Update state
            block_idx = block_triple_indices[best_idx]
            newly_covered = np.sum(uncovered_mask[block_idx])
            uncovered_mask[block_idx] = False
            remaining -= newly_covered
            selected_blocks.append(best_idx)
            
        return selected_blocks

    def reduce_covering(block_indices):
        """Remove redundant blocks."""
        if not block_indices:
            return []
        
        # Create a set of selected block indices for fast lookup? 
        # Or just iterate.
        # For each block, check if all its triples are covered by OTHER selected blocks.
        # This is O(n_blocks_selected * k_t * n_blocks_selected) if naive.
        # Better: 
        # 1. Build a coverage count array for all triples based on selected blocks.
        # 2. For each selected block, check if removing it leaves any triple with count 0.
        
        n_selected = len(block_indices)
        if n_selected == 0:
            return []
        
        # Count coverage
        coverage_counts = np.zeros(n_triples, dtype=np.int32)
        for b_idx in block_indices:
            coverage_counts[block_triple_indices[b_idx]] += 1
            
        essential_blocks = []
        for b_idx in block_indices:
            # Check if this block is essential
            # If any triple in this block has coverage_count == 1, it's essential.
            # If all have count > 1, it's redundant.
            block_triples = block_triple_indices[b_idx]
            if np.any(coverage_counts[block_triples] == 1):
                essential_blocks.append(b_idx)
            else:
                # It's redundant. We could remove it and update counts, 
                # but for simplicity in a single pass reduction, we just keep the essential ones.
                # Note: Removing one redundant block might make another redundant.
                # A single pass is a good heuristic.
                pass
        
        return essential_blocks

    # Run multiple times with different seeds to find a better covering
    best_blocks = None
    best_len = float('inf')
    
    # Increase restarts slightly, but keep time in check.
    # Precomputation is the bottleneck.
    # C(24,7) = 346104 blocks.
    # Filling block_triple_indices takes time.
    
    n_restarts = 30
    
    for i in range(n_restarts):
        raw_blocks = greedy_cover(i, top_k=20)
        reduced_blocks = reduce_covering(raw_blocks)
        
        # Convert back to lists of points
        final_blocks = [list(all_blocks[idx]) for idx in reduced_blocks]
        
        if len(final_blocks) < best_len:
            best_len = len(final_blocks)
            best_blocks = final_blocks
            
    return best_blocks
