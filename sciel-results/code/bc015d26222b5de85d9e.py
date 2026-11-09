def _precompute_masks_for_residue(r, n=12, max_L=144):
    masks = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        # j such that (r+j) % i == 0
        # j % i == (-r) % i
        rem = (-r) % i
        # j = rem, rem+i, rem+2i, ...
        # j must be in 1..max_L
        # If rem == 0, j = i, 2i, ... (since j>=1)
        # If rem != 0, j = rem, rem+i, ...
        
        start = rem
        if start == 0:
            start = i
            
        j = start
        while j <= max_L:
            mask |= (1 << (j - 1))
            j += i
        masks[i] = mask
    return masks
