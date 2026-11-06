"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find the minimal start s (1 <= s <= 1004) such that s, s+1, s+2, s+3, s+4
    are all non-zero quadratic residues modulo 1009, and return the start and
    one root for each residue.
    
    Approach: Use the fact that 1009 is prime and compute the set of quadratic
    residues efficiently. Then scan from s=1 upwards to find the first valid run.
    """
    p = 1009
    # Compute all quadratic residues mod p and store one root for each
    # We need non-zero residues only (1..1008)
    root_for_residue = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:  # Exclude zero residue
            if r not in root_for_residue:
                root_for_residue[r] = x
    
    # Now scan from s=1 to s=1004 to find the minimal s
    # such that s, s+1, s+2, s+3, s+4 are all in root_for_residue
    # Note: s+4 must be <= 1008 since residues are 1..1008
    # So s can be at most 1004 (since s+4 <= 1008)
    
    for s in range(1, 1005):
        # Check if s, s+1, s+2, s+3, s+4 are all valid non-zero QRs
        valid = True
        for i in range(5):
            val = s + i
            if val > 1008 or val not in root_for_residue:
                valid = False
                break
        if valid:
            roots = [root_for_residue[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}
    
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
