"""Find the minimum start s for 5 consecutive quadratic residues mod 1009."""

def construct():
    p = 1009
    
    # Precompute roots: residue -> a root x (1 <= x <= p-1) such that x^2 mod p == residue
    # Only non-zero residues are valid; zero is excluded.
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r == 0:
            continue
        if r not in roots:
            roots[r] = x
    
    # We need to find the smallest s (1 <= s <= p-5 = 1004) such that
    # s, s+1, s+2, s+3, s+4 are all in roots (i.e., all non-zero quadratic residues).
    # Note: s+i must be <= 1008, so s <= 1004.
    
    # Simple ascending search from s=1
    for s in range(1, p - 4):
        if all((s + i) in roots for i in range(5)):
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
    
    # Should not reach here, but just in case
    raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found')
