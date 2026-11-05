"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Precompute: for each residue r in 1..p-1, store a root x such that x^2 % p == r
    # We only care about residues in 1..p-1 (zero excluded)
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and r not in roots:
            roots[r] = x
    
    # Search for the smallest s in 1..p-5 such that s, s+1, s+2, s+3, s+4 are all quadratic residues
    # (all in 1..p-1, which is automatic since s>=1 and s+4 <= p-1=1008 when s <= p-5=1004)
    for s in range(1, p - 4):
        if (s in roots and 
            (s+1) in roots and 
            (s+2) in roots and 
            (s+3) in roots and 
            (s+4) in roots):
            return {'start': s, 'roots': [roots[s+i] for i in range(5)]}
    
    raise RuntimeError('No run of 5 consecutive quadratic residues found')
# EVOLVE-BLOCK-END
