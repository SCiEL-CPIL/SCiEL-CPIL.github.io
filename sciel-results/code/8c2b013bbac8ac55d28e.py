def construct():
    p = 1009
    
    # Initialize bitset for residues and a map for roots
    # bit i is 1 if i is a non-zero quadratic residue mod p
    # root_map[i] stores one root x such that x^2 % p == i
    bitset = 0
    root_map = {}
    
    # Generate all non-zero quadratic residues and their roots
    # Since (p-x)^2 == x^2 mod p, we can iterate x from 1 to p//2
    # but iterating 1 to p-1 is simple enough and ensures we find roots in 1..1008
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            # Set the bit for residue r
            bitset |= (1 << r)
            # Store the first root found for this residue
            # We want roots in 1..1008, which x is.
            if r not in root_map:
                root_map[r] = x
                
    # The mask for checking 5 consecutive bits
    mask = 0b11111
    
    # Find the smallest s (1 <= s <= 1004) such that s..s+4 are all residues
    # We iterate s from 1 upwards. The first valid s is the minimum.
    for s in range(1, 1005):
        # Check if bits s, s+1, s+2, s+3, s+4 are all set in the bitset
        # (bitset >> s) shifts the bits so that bit s becomes bit 0.
        # Then we check if the lowest 5 bits are all 1.
        if (bitset >> s) & mask == mask:
            # Found the minimal start s
            # Construct the roots list for s, s+1, s+2, s+3, s+4
            roots = [root_map[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}
            
    # This should not happen for p=1009 and m=5
    raise RuntimeError("No run of 5 consecutive non-zero quadratic residues found")
