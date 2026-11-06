"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Precompute quadratic residues and one root for each
    # roots[x] = r such that r*r % p == x
    roots = {}
    for r in range(1, p):
        x = (r * r) % p
        # We just need one root. Since we iterate r from 1 to p-1,
        # the first one found is the smallest positive root.
        # The problem doesn't specify which root, just that ri**2 mod p == s+i.
        if x not in roots:
            roots[x] = r
    
    # Search for the minimum s
    # s must be in [1, 1004] so that s+4 <= 1008
    for s in range(1, 1005):
        # Check if s, s+1, s+2, s+3, s+4 are all quadratic residues
        # and non-zero (which is guaranteed by s >= 1 and s+4 <= 1008)
        if all((s + i) in roots for i in range(5)):
            res_roots = [roots[s + i] for i in range(5)]
            return {'start': s, 'roots': res_roots}
            
    # Fallback, though should not happen
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
