<<<<<<< SEARCH
    # For valid pairs, calculate heuristic
    # We add a small epsilon to avoid division by zero for perfect fits (remaining=0)
    # and to ensure values are positive.
    eps = 1e-9
    valid_remaining = remaining[valid_mask] + eps
    
    # Calculate 1 / remaining
    # This results in very high values for tight fits and low values for loose fits.
    h[valid_mask] = 1.0 / valid_remaining
=======
    # For valid pairs, calculate heuristic
    # We add a small epsilon to avoid division by zero for perfect fits (remaining=0)
    # and to ensure values are positive.
    eps = 1e-9
    valid_remaining = remaining[valid_mask] + eps
    
    # Weight by size of item j: larger items are preferred to fill remaining space
    # This encourages packing larger items first, which typically reduces bin count
    item_j_sizes = d[None, :]  # shape (1, n) for broadcasting
    valid_item_j = item_j_sizes[valid_mask]
    
    # Calculate size(j) / remaining
    # This results in high values for tight fits with large items
    h[valid_mask] = valid_item_j / valid_remaining
>>>>>>> REPLACE
