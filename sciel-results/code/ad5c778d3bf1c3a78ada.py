<<<<<<< SEARCH
def _reduce_blocks(v, k, t, block_indices, block_masks, full_mask):
    if not block_indices:
        return []
        
    current_blocks = list(block_indices)
    changed = True
    
    while changed:
        changed = False
        new_current = []
        n = len(current_blocks)
        
        # Pre-compute union of all? 
        # If we remove block i, the union of others is (total_union ^ block_masks[i]) only if block i is unique?
        # No, XOR doesn't work for union if bits overlap.
        # We need OR of all others.
        # Since N is small (~76), O(N^2) is fine.
        
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
=======
def _reduce_blocks(v, k, t, block_indices, block_masks, full_mask):
    if not block_indices:
        return []
        
    current_blocks = list(block_indices)
    changed = True
    
    while changed:
        changed = False
        new_current = []
        n = len(current_blocks)
        
        # Pre-compute total union
        total_union = 0
        for idx in current_blocks:
            total_union |= block_masks[idx]
        
        for i, idx in enumerate(current_blocks):
            # Check if block i is redundant
            # Union of all blocks except i = total_union if block i's unique bits are covered by others
            # More precisely: if (total_union & ~block_masks[idx]) still covers full_mask
            union_others = total_union & ~block_masks[idx]
            
            if (union_others & full_mask) == full_mask:
                # Redundant
                changed = True
            else:
                new_current.append(idx)
                
        current_blocks = new_current
        
    return current_blocks
>>>>>>> REPLACE

