import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using randomized greedy set cover
    with local search (redundancy removal).
    """
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # Pre-compute all t-subsets (triples) and assign them indices
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Pre-compute all k-subsets (blocks) and their corresponding triple indices
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map each block to the indices of the t-subsets it contains
    block_triple_indices = np.empty((num_blocks, k * (k-1) // (t*(t-1)) if t>1 else k), dtype=np.int32)
    # Correction: number of t-subsets in a k-set is C(k, t)
    num_t_in_k = 1
    if t > 0:
        num_t_in_k = 1
        for i in range(t):
            num_t_in_k *= (k - i)
            num_t_in_k //= (i + 1)
            
    block_triple_indices = np.empty((num_blocks, num_t_in_k), dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        block_set = set(block)
        indices = []
        # Generate all t-subsets of this block
        for sub in itertools.combinations(block, t):
            indices.append(triple_to_idx[sub])
        block_triple_indices[i] = indices

    best_blocks = None
    best_size = float('inf')
    
    # Parameters for the search
    num_restarts = 50
    top_k_candidates = 10
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Randomized Greedy Set Cover
        uncovered = np.ones(num_triples, dtype=np.int8)
        covered_count = np.zeros(num_triples, dtype=np.int32)
        current_blocks = []
        
        # We need to track which blocks are already chosen to avoid duplicates if we were doing something else, 
        # but in standard greedy we just pick the best one. 
        # However, picking the same block again is useless. 
        # Since we update 'uncovered', the gain of a chosen block becomes 0 (if it was fully uncovered) 
        # or stays low. But to be safe and efficient, we can just rely on the gain.
        # Actually, if a block is chosen, its triples are set to 0 in 'uncovered'. 
        # So its gain will be 0 in the next iteration (unless it has other triples, but it doesn't).
        # Wait, a block covers specific triples. Once chosen, those triples are covered.
        # The gain is sum(uncovered[indices]). So chosen blocks will have gain 0.
        
        while np.any(uncovered):
            if time.time() - start_time > time_limit - 5:
                break
                
            # Calculate gains
            # uncovered is (num_triples,), block_triple_indices is (num_blocks, num_t_in_k)
            # This operation creates a (num_blocks, num_t_in_k) array temporarily
            gains = uncovered[block_triple_indices].sum(axis=1)
            
            # Find top K candidates
            # To add randomness, we can pick randomly from the top K
            if np.max(gains) == 0:
                # Should not happen if uncovered exists, but safety
                break
            
            # Get indices of top K
            # argsort is O(N log N). 350k log 350k is fast enough?
            # Alternatively, use argpartition for O(N)
            top_indices = np.argpartition(gains, -top_k_candidates)[-top_k_candidates:]
            
            # Randomly select one from the top K
            # Weighted random or just uniform? Uniform is fine for diversity.
            chosen_idx = top_indices[random.randint(0, top_k_candidates - 1)]
            
            current_blocks.append(all_blocks[chosen_idx])
            
            # Update uncovered and covered_count
            block_triples = block_triple_indices[chosen_idx]
            uncovered[block_triples] = 0
            covered_count[block_triples] += 1
            
        if not np.any(uncovered):
            # Valid covering found. Apply local search (redundancy removal)
            # covered_count currently holds the count from this specific run.
            # We can use it directly.
            
            # Redundancy Removal
            # A block is redundant if all its triples have count >= 2
            # We iterate and remove blocks if possible.
            # To be efficient, we can just loop through the list.
            
            # Sort blocks? No, just iterate.
            # We need to be careful: removing a block decreases counts.
            
            blocks_list = list(current_blocks)
            
            # Optimization: Check for redundant blocks in a loop
            changed = True
            while changed:
                changed = False
                # Iterate in reverse to allow easy removal? Or just mark for removal.
                # Marking is safer.
                to_remove = []
                for i, block in enumerate(blocks_list):
                    block_triples = block_triple_indices[all_blocks.index(block)] # This is slow! O(N)
                    # We should have stored the index in current_blocks
                    # Let's refactor to store indices.
                    pass
                    
            # Refactor: Store indices in current_blocks
            # Let's restart the inner loop logic with indices
            
            # Actually, let's just re-run the redundancy check on the final list using precomputed indices
            # We need to map block tuple -> index.
            block_to_idx = {b: i for i, b in enumerate(all_blocks)}
            
            blocks_indices = [block_to_idx[b] for b in current_blocks]
            
            # Re-calculate covered_count from scratch for the reduced set? 
            # The covered_count from the greedy phase is correct for the initial set.
            # As we remove blocks, we update it.
            
            # Current covered_count is for the full greedy set.
            # We will simulate removal.
            
            # To speed up, we can just check redundancy.
            # For each block, check if min(covered_count[block_triples]) >= 2.
            
            active = [True] * len(blocks_indices)
            
            # Iterate until no change
            while True:
                removed_any = False
                for i in range(len(blocks_indices)):
                    if not active[i]:
                        continue
                    idx = blocks_indices[i]
                    block_triples = block_triple_indices[idx]
                    # Check if all triples in this block are covered by others
                    # i.e., count >= 2
                    if np.min(covered_count[block_triples]) >= 2:
                        # Remove it
                        active[i] = False
                        removed_any = True
                        # Update counts
                        covered_count[block_triples] -= 1
                        if time.time() - start_time > time_limit - 2:
                            break
                if not removed_any or time.time() - start_time > time_limit - 2:
                    break
            
            final_blocks = [all_blocks[blocks_indices[i]] for i in range(len(blocks_indices)) if active[i]]
            
            if len(final_blocks) < best_size:
                best_size = len(final_blocks)
                best_blocks = final_blocks
                
    if best_blocks is None:
        # Fallback: return a simple greedy result if time ran out during search
        # Or just return the first valid one found? 
        # If no valid one found, we must return something valid.
        # The greedy loop usually finds one. If time runs out in the middle, 
        # we might have an incomplete cover. 
        # The task says "Valid when every t-subset...". 
        # If we can't finish, we should probably return a large valid cover.
        # But the greedy loop is fast. The risk is in the redundancy loop or multiple restarts.
        # If best_blocks is None, it means we didn't complete a single valid cover?
        # Unlikely. 
        # To be safe, if best_blocks is None, we do a single non-randomized greedy without time check on reduction.
        uncovered = np.ones(num_triples, dtype=np.int8)
        current_blocks = []
        while np.any(uncovered):
            gains = uncovered[block_triple_indices].sum(axis=1)
            chosen_idx = np.argmax(gains)
            current_blocks.append(all_blocks[chosen_idx])
            block_triples = block_triple_indices[chosen_idx]
            uncovered[block_triples] = 0
            
        return current_blocks
        
    return best_blocks
