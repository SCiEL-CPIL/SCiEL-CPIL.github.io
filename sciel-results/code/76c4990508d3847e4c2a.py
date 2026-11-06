"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find the minimal start s (1 <= s <= 1004) such that s, s+1, s+2, s+3, s+4
    are all non-zero quadratic residues modulo 1009, and return the start and
    one root for each residue.
    
    Approach: Use a lookup table for roots to enable O(1) checks.
    """
    p = 1009
    # roots[r] stores a root x such that x^2 % p == r, or 0 if r is not a QR (or r==0)
    roots = [0] * p
    
    for x in range(1, p):
        r = (x * x) % p
        # We only care about non-zero residues for the run, but we store roots for all non-zero x
        # If r is already set, we can keep the first one or any one.
        # The task asks for *a* root.
        if r != 0:
            if roots[r] == 0:
                roots[r] = x

    # Scan for minimal s
    for s in range(1, 1005):
        # Check s, s+1, s+2, s+3, s+4
        # Note: s+4 <= 1008 is guaranteed by s <= 1004
        valid = True
        for i in range(5):
            val = s + i
            if roots[val] == 0:
                valid = False
                break
        
        if valid:
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
    
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
