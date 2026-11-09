"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Precompute a set of quadratic residues modulo p (excluding 0)
    residues = set()
    roots = {}
    for x in range(1, p):
        v = (x * x) % p
        residues.add(v)
        if v not in roots:
            roots[v] = x

    # We want the smallest s in [1, 1004] such that s, s+1, s+2, s+3, s+4 are all in residues
    # and s+4 <= 1008 (since residues are in [1, 1008])
    # Actually s+i must be <= 1008 for i in 0..4, so s <= 1004
    # Also s >= 1
    
    # Since we want to minimize s, search from s=1 upward
    for s in range(1, 1005):
        ok = True
        for i in range(5):
            val = s + i
            if val > 1008 or val not in residues:
                ok = False
                break
        if ok:
            return {'start': s, 'roots': [roots[s+i] for i in range(5)]}
    
    raise RuntimeError('No baseline run found')
# EVOLVE-BLOCK-END
