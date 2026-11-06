"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    m = 5
    
    # Precompute: for each residue r in 1..p-1, store a root x such that x*x % p == r
    # We only care about residues that are quadratic residues (non-zero)
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and r not in roots:
            roots[r] = x
    
    # Create a boolean array for quick lookup: is_residue[r] = True if r is a quadratic residue
    is_residue = [False] * p
    for r in roots:
        is_residue[r] = True
    
    # Ascending search for the minimum start s
    # s ranges from 1 to p-1-m+1 = 1009-5+1 = 1005, but constraint says 1<=s<=1004
    # so s+i for i in 0..4 means s+4 <= 1008, so s <= 1004. Correct.
    for s in range(1, 1005):
        # Check if all residues s, s+1, s+2, s+3, s+4 are quadratic residues
        all_found = True
        for i in range(m):
            if not is_residue[s + i]:
                all_found = False
                break
        if all_found:
            return {'start': s, 'roots': [roots[s + i] for i in range(m)]}
    
    # If no run found (should not happen for p=1009), raise error
    raise RuntimeError('No run of 5 consecutive quadratic residues found')
# EVOLVE-BLOCK-END
