"""Find the minimum start s for 5 consecutive quadratic residues mod 1009."""

import time

def construct():
    p = 1009
    deadline = time.time() + 80  # safety margin before 90s limit
    
    # Precompute: for each residue r in [1, p-1], store a root x such that x^2 % p == r
    # We only care about non-zero residues (1 <= r <= p-1)
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and r not in roots:
            roots[r] = x
    
    # Build a boolean array: is_residue[r] = True if r is a non-zero quadratic residue mod p
    is_residue = [False] * p
    for r in roots:
        is_residue[r] = True
    
    # Search for the smallest s such that s, s+1, s+2, s+3, s+4 are all non-zero quadratic residues
    # s must be in [1, p-5] = [1, 1004]
    # Also s+i must be in [1, p-1] for all i in 0..4, so s <= p-5 = 1004
    # And s+i >= 1, so s >= 1
    
    best_result = None
    for s in range(1, p - 4):  # s from 1 to 1004
        if time.time() > deadline:
            break
        # Check if s, s+1, s+2, s+3, s+4 are all quadratic residues
        valid = True
        for i in range(5):
            val = s + i
            if val >= p or not is_residue[val]:
                valid = False
                break
        if valid:
            best_result = {'start': s, 'roots': [roots[s+i] for i in range(5)]}
            break
    
    if best_result is None:
        raise RuntimeError('No run of 5 consecutive quadratic residues found')
    
    return best_result
