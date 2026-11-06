"""Find the minimum start s for 5 consecutive quadratic residues mod 1009."""

def construct():
    p = 1009
    
    # Precompute which residues are non-zero quadratic residues
    # and store one root for each.
    # We need: for each residue r in [1, p-1], a root x in [1, p-1] such that x*x % p == r.
    # Zero residues (r=0) are excluded.
    
    is_residue = [False] * p
    root_for = [0] * p  # root_for[r] = x such that x*x % p == r, for r in 1..p-1
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            if not is_residue[r]:
                is_residue[r] = True
                root_for[r] = x
    
    # Search for the smallest s such that s, s+1, s+2, s+3, s+4 are all non-zero quadratic residues
    # s must be in [1, 1004] so that s+4 <= 1008 = p-1 (no wraparound)
    for s in range(1, p - 4):  # s from 1 to 1004 inclusive
        if all(is_residue[s + i] for i in range(5)):
            return {
                'start': s,
                'roots': [root_for[s + i] for i in range(5)]
            }
    
    raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found')
