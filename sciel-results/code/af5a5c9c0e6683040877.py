import time

def construct():
    p = 1009
    m = 5
    
    # Precompute roots for each non-zero quadratic residue
    # We want the smallest positive root for each residue
    roots_map = {}
    for x in range(1, p):
        r = (x * x) % p
        if r not in roots_map:
            roots_map[r] = x
            
    # Build bitset for quadratic residues
    # Bit i is 1 if i is a non-zero quadratic residue
    res = 0
    for r, root in roots_map.items():
        res |= (1 << r)
        
    # We need s, s+1, s+2, s+3, s+4 to all be residues.
    # This is equivalent to finding the lowest set bit in:
    # res & (res >> 1) & (res >> 2) & (res >> 3) & (res >> 4)
    # Let's verify the shift direction.
    # If bit s is 1 in (res >> 1), then bit s+1 is 1 in res. Correct.
    
    mask = res & (res >> 1) & (res >> 2) & (res >> 3) & (res >> 4)
    
    if mask == 0:
        # Fallback to linear search if bitset logic fails (should not happen)
        for s in range(1, p - m + 1):
            valid = True
            for i in range(m):
                if (s + i) not in roots_map:
                    valid = False
                    break
            if valid:
                return {'start': s, 'roots': [roots_map[s+i] for i in range(m)]}
        raise RuntimeError("No run found")
        
    # Find the index of the lowest set bit
    # mask & -mask isolates the lowest set bit
    lowest_bit = mask & -mask
    s = lowest_bit.bit_length() - 1
    
    # Verify s is within bounds
    # The bitset construction ensures s+4 <= 1008 because bits > 1008 are 0
    # and s >= 1 because bit 0 of res is 0.
    
    res_roots = [roots_map[s + i] for i in range(m)]
    
    return {'start': s, 'roots': res_roots}
