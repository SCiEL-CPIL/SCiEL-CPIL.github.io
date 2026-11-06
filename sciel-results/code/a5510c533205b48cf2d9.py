"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Compute all quadratic residues mod p and store one root for each
    # We need ri in range [1, 1008], and ri^2 % p == residue
    # Since p is prime, for each non-zero residue there are exactly 2 roots
    # We'll store the smaller root for efficiency (though any valid root works)
    
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r == 0:
            continue  # exclude zero residues
        # Store the first (smallest) root we find for this residue
        if r not in roots:
            roots[r] = x
    
    # Scan from s=1 upward to find the minimal s such that
    # s, s+1, s+2, s+3, s+4 are all non-zero quadratic residues
    # Note: s+i must be in [1, 1008] for i=0..4, so s can be at most 1004
    for s in range(1, 1005):
        # Check if all five consecutive values are quadratic residues
        found = True
        for i in range(5):
            val = s + i
            if val > 1008:
                found = False
                break
            if val not in roots:
                found = False
                break
        if found:
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
    
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
