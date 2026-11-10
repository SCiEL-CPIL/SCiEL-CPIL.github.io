"""Rewritten program using a direct, optimized search for the minimum run-start."""
import time

# EVOLVE-BLOCK-START
def construct():
    p = 1009
    m = 5
    
    # Precompute the smallest root for each non-zero quadratic residue modulo p.
    # Use a list for O(1) access without hash overhead.
    # root_of[r] = smallest x such that x*x % p == r, or 0 if r is not a residue.
    root_of = [0] * p
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and root_of[r] == 0:
            root_of[r] = x

    # We want the smallest s in [1, 1004] such that s, s+1, s+2, s+3, s+4 are all in roots.
    # Iterate s from 1 upward.
    # s can go up to 1004 because s+4 <= 1008.
    for s in range(1, p - m + 1):
        # Check if s, s+1, ..., s+4 are all quadratic residues (non-zero)
        # root_of[val] != 0 implies val is a non-zero quadratic residue.
        if (root_of[s] != 0 and 
            root_of[s+1] != 0 and 
            root_of[s+2] != 0 and 
            root_of[s+3] != 0 and 
            root_of[s+4] != 0):
            return {'start': s, 'roots': [root_of[s+i] for i in range(m)]}

    # Fallback: if somehow not found (shouldn't happen)
    raise RuntimeError('No run found')
# EVOLVE-BLOCK-END
