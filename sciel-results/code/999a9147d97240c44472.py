"""Rewritten program: find the minimal start s such that s..s+4 are all quadratic residues mod 1009."""
import time

def construct():
    p = 1009
    # Precompute a mapping from residue to one root (smallest root)
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            if r not in roots or x < roots[r]:
                roots[r] = x
    
    # Build a boolean list: is_qr[v] is True if v (1..1008) is a nonzero QR mod p
    is_qr = [False] * (p + 1)
    for r in roots:
        is_qr[r] = True
    
    # Find minimal s in 1..1004 such that s..s+4 are all QR
    # s+i <= 1004+4 = 1008, so no wraparound to p=1009
    for s in range(1, 1005):
        valid = True
        for i in range(5):
            if not is_qr[s + i]:
                valid = False
                break
        if valid:
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
    
    raise RuntimeError('No valid run found')
