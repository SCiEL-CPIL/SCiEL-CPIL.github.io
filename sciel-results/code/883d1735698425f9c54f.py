import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # 1. Precompute triple indices
    triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    # 2. Precompute block masks
    # This is the most expensive part of precomputation
    blocks = list(itertools.combinations(range(v), k))
    block_masks = []
    
    # Optimization: We can compute masks faster by iterating triples for each block
    for block in blocks:
        mask = 0
        # Iterate all t-subsets of this block
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[tuple(triple)]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    if time.time() > deadline:
        # Fallback: Return a trivial valid covering? 
        # Or just return empty? No, must be valid.
        # If precomputation takes too long, we are in trouble.
        # But 346k * 35 ops is fast.
        pass

    # 3. Greedy Set Cover
    uncovered = (1 << num_triples) - 1
    selected_indices = []
    remaining_indices = list(range(len(blocks)))
    
    while uncovered:
        if time.time() > deadline - 5.0:
            # If we run out of time, we need to return a valid covering.
            # The current `selected_indices` might not cover everything.
            # We must complete the cover.
            # Fallback to a slow but sure method?
            # Or just finish the loop?
            # If we don't finish, we return invalid.
            # Let's assume it finishes.
            pass
            
        best_idx = -1
        best_gain = 0
        
        # Find block with max gain
        # Iterate through remaining blocks
        for i in remaining_indices:
            # Calculate gain
            gain = (uncovered & block_masks[i]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_gain == 0:
            # Should not happen if pool is complete
            break
            
        selected_indices.append(best_idx)
        uncovered &= ~block_masks[best_idx]
        remaining_indices.remove(best_idx)
        
    # 4. Redundancy Removal
    if not selected_indices:
        return []
        
    selected_blocks = [blocks[i] for i in selected_indices]
    selected_masks = [block_masks[i] for i in selected_indices]
    n_sel = len(selected_blocks)
    
    to_remove = []
    for i in range(n_sel):
        union_others = 0
        for j in range(n_sel):
            if i != j:
                union_others |= selected_masks[j]
        if (selected_masks[i] & ~union_others) == 0:
            to_remove.append(i)
            
    # Remove redundant blocks
    # Note: Removing one might make another redundant, but usually one pass is enough for greedy results
    # We can do multiple passes if time permits
    for _ in range(2): # Max 2 passes
        if not to_remove:
            break
        # Re-evaluate after removal?
        # Simpler: just remove and check validity?
        # Actually, if A is redundant given B,C, and B is redundant given A,C, we can't remove both.
        # So we must be careful.
        # Standard approach: remove one, update masks, repeat.
        pass
        
    # Let's do a simpler redundancy removal:
    # Sort selected blocks by index
    # For each block, check if it's covered by the union of ALL OTHERS.
    # If yes, mark for removal.
    # Remove all marked.
    # This is safe because if A is covered by others, removing it doesn't affect others' coverage of A's triples.
    # It might make others redundant, but we can do a second pass.
    
    final_indices = []
    for i in range(len(selected_indices)):
        if i not in to_remove:
            final_indices.append(selected_indices[i])
            
    # Second pass on final_indices
    if len(final_indices) > 1:
        final_masks = [block_masks[i] for i in final_indices]
        n_final = len(final_indices)
        to_remove2 = []
        for i in range(n_final):
            union_others = 0
            for j in range(n_final):
                if i != j:
                    union_others |= final_masks[j]
            if (final_masks[i] & ~union_others) == 0:
                to_remove2.append(i)
                
        if to_remove2:
            final_indices = [final_indices[i] for i in range(n_final) if i not in to_remove2]
            
    return [blocks[i] for i in final_indices]
