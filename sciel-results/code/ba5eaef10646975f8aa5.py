"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    deadline = time.time() + 60.0
    
    # Precompute: for each residue r in [1, p-1], store one root x in [1, p-1]
    # such that x*x % p == r. We want to minimize s, so we just need any valid root.
    roots = {}
    for x in range(1, p):
        v = (x * x) % p
        if v == 0:
            continue  # zero residues excluded
        if v not in roots:
            roots[v] = x
    
    # Forward search: find the smallest s in [1, p-5] such that
    # s, s+1, s+2, s+3, s+4 are all quadratic residues (non-zero).
    # Note: s+i must be in [1, p-1] for all i in [0,4], so s <= p-5 = 1004.
    for s in range(1, p - 4):  # s from 1 to 1004
        if s + 4 >= p:
            break
        # Check if all five consecutive values are non-zero quadratic residues
        ok = True
        for i in range(5):
            val = s + i
            if val >= p or val == 0 or val not in roots:
                ok = False
                break
        if ok:
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
    
    # Fallback: descending search (should not be reached)
    for s in range(p - 5, 0, -1):
        if all(1 <= s + i < p and (s + i) in roots for i in range(5)):
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
    
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
