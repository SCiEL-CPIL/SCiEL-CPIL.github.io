def construct():
    p = 1009
    # roots[i] will store one root x such that x^2 == i mod p
    # We use 0 to indicate "not a residue" or "no root stored yet"
    # Since roots must be 1..1008, 0 is a safe sentinel for "none".
    roots = [0] * p
    
    for x in range(1, p):
        r = (x * x) % p
        # Store the first root found. Since we iterate x from 1, 
        # this will be the smaller of the two roots (x, p-x) if x < p/2,
        # or just a valid root. It doesn't matter which valid root we pick.
        if roots[r] == 0:
            roots[r] = x
            
    # Build a bitset where bit i is 1 if i is a non-zero quadratic residue
    # We need bits 1 through 1008 to be meaningful
    # Use an integer as a bitset
    bitset = 0
    for i in range(1, p):
        if roots[i] != 0:
            bitset |= (1 << i)
    
    # Search for the smallest s such that bits s, s+1, s+2, s+3, s+4 are all set
    # This is equivalent to checking if (bitset >> s) & 0b11111 == 0b11111
    # s must be at least 1 and s+4 must be at most 1008, so s in [1, 1004]
    for s in range(1, 1005):
        # Check if 5 consecutive bits starting at position s are all set
        if (bitset >> s) & 0b11111 == 0b11111:
            # Extract the roots
            res_roots = [roots[s+i] for i in range(5)]
            return {'start': s, 'roots': res_roots}
            
    # This should never happen for p=1009, but just in case
    raise RuntimeError('No run of 5 consecutive quadratic residues found')
