"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Build a bitmask where bit v is set iff v is a non-zero quadratic residue mod 1009
    # We need residues in range 1..1008 (non-zero)
    # For each x in 1..1008, compute x*x % 1009; if non-zero, set that bit
    # Also store the root for each residue (the first x that produces it)
    
    p = 1009
    roots = {}
    mask = 0
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            mask |= (1 << r)
            if r not in roots:
                roots[r] = x
    
    # Now search for the smallest s in 1..1004 such that s, s+1, s+2, s+3, s+4
    # are all non-zero quadratic residues.
    # Note: s+i must be in 1..1008, so s <= 1004 ensures s+4 <= 1008.
    # Check: (mask >> s) & 0x1F == 0x1F means bits s through s+4 are all set.
    
    for s in range(1, 1005):
        # Check if bits s, s+1, s+2, s+3, s+4 are all set in mask
        if (mask >> s) & 0x1F == 0x1F:
            return {'start': s, 'roots': [roots[s+i] for i in range(5)]}
    
    raise RuntimeError('No baseline run found')
# EVOLVE-BLOCK-END
