def construct():
    p = 1009
    
    # Approach: Bitmask representation of quadratic residues
    # mask bit i is 1 if i is a non-zero quadratic residue mod p
    mask = 0
    # Precompute a root for each residue
    # We use a list where index is the residue, value is the root
    # Initialize with 0 (which is not a valid root index in 1..p-1 for our purposes, 
    # and 0 is excluded as a residue anyway)
    roots = [0] * p
    
    for x in range(1, p):
        r = (x * x) % p
        if r == 0:
            continue
        # Set the bit in mask
        mask |= (1 << r)
        # Store the smallest root (since we iterate x from 1 up)
        if roots[r] == 0:
            roots[r] = x
            
    # Find smallest s such that s..s+4 are all residues
    # We check if the 5 bits starting at s are all 1
    target_pattern = 0b11111
    
    for s in range(1, p - 4): # s can be up to p-5 = 1004
        # Shift mask right by s, check if lowest 5 bits are all 1
        if (mask >> s) & target_pattern == target_pattern:
            res_roots = [roots[s + i] for i in range(5)]
            return {'start': s, 'roots': res_roots}
            
    raise RuntimeError("No valid run found")
