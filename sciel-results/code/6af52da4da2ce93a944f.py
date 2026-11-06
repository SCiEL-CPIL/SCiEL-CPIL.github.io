"""Find the least run-start s for 5 consecutive non-zero quadratic residues mod 1009."""

def construct():
    P = 1009
    M = 5
    
    # Build a map from each non-zero quadratic residue to one valid root
    roots = {}
    for x in range(1, P):
        r = (x * x) % P
        if r != 0 and r not in roots:
            roots[r] = x
    
    # Ascending search to find the minimum s
    for s in range(1, P - M + 1):
        # Check that s, s+1, ..., s+4 are all in roots (non-zero quadratic residues)
        if all((s + i) in roots for i in range(M)):
            return {'start': s, 'roots': [roots[s + i] for i in range(M)]}
    
    raise RuntimeError('No valid run found')
