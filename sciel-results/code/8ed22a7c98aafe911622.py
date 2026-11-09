"""Rewritten program to find the minimal start of a run of 5 quadratic residues."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    m = 5
    
    # Precompute the smallest root for each non-zero quadratic residue
    # roots[r] = x such that x*x % p == r
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and r not in roots:
            roots[r] = x
            
    # Search for the smallest s such that s, s+1, ..., s+m-1 are all in roots
    # s must be at least 1 and at most p - m to stay within 1..p-1 without wrap-around issues 
    # specifically s+i <= 1008 implies s <= 1004
    for s in range(1, p - m + 1):
        if all((s + i) in roots for i in range(m)):
            return {'start': s, 'roots': [roots[s + i] for i in range(m)]}
            
    # This should never happen for p=1009, but just in case
    raise RuntimeError("No run of 5 quadratic residues found")
# EVOLVE-BLOCK-END
