"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""

# EVOLVE-BLOCK-START
def construct():
    p = 1009
    
    # Precompute roots for each residue
    roots = [0] * p
    
    for x in range(1, p):
        r = (x * x) % p
        if roots[r] == 0:
            roots[r] = x
    
    # Build a bitmask where bit i is set iff i is a non-zero quadratic residue
    # We need bits 1..1008 (0 is excluded)
    # Use Python's arbitrary precision integers for bitwise operations
    mask = 0
    for i in range(1, p):
        if roots[i] != 0:
            mask |= (1 << i)
    
    # Find the smallest s such that s, s+1, ..., s+4 are all QR
    # We check: (mask >> s) & 0b11111 == 0b11111
    # This means bits s through s+4 are all set
    for s in range(1, 1005):
        if (mask >> s) & 0b11111 == 0b11111:
            res_roots = [roots[s + i] for i in range(5)]
            return {'start': s, 'roots': res_roots}
    
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
