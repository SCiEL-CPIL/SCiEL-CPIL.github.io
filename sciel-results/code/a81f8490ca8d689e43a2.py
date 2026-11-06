"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find the minimal start s (1 <= s <= 1004) such that s, s+1, s+2, s+3, s+4
    are all non-zero quadratic residues modulo 1009, and return the start and
    one root for each residue.
    
    Approach: Use a list to store roots indexed by residue value for O(1) 
    access with lower overhead than a dictionary. Scan for the first valid run.
    """
    p = 1009
    # Initialize list with None to indicate "not a quadratic residue" or "no root found yet"
    # We only care about non-zero residues, so index 0 will remain None/unused for valid checks
    roots = [None] * p
    
    # Compute roots for all non-zero quadratic residues
    for x in range(1, p):
        r = (x * x) % p
        if roots[r] is None:
            roots[r] = x
            
    # Scan for the minimal start s
    # s can range from 1 to p-5 (which is 1004) because s+4 must be <= p-1 (1008)
    for s in range(1, p - 4):
        # Check if s, s+1, s+2, s+3, s+4 all have roots
        if roots[s] is not None and \
           roots[s+1] is not None and \
           roots[s+2] is not None and \
           roots[s+3] is not None and \
           roots[s+4] is not None:
            
            return {'start': s, 'roots': [roots[s], roots[s+1], roots[s+2], roots[s+3], roots[s+4]]}
    
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
