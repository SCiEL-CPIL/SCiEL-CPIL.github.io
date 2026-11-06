"""Find the minimum start s for 5 consecutive quadratic residues mod 1009."""

import time

def construct():
    p = 1009
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    # Precompute roots: map each non-zero quadratic residue to a root x in [1, p-1]
    # such that x*x % p == residue.
    # We only need one root per residue.
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and r not in roots:
            roots[r] = x
    
    # Build a bitset where bit i is 1 if i is a non-zero quadratic residue mod p
    bitset = 0
    for r in roots:
        bitset |= (1 << r)
    
    # Create a mask that has 1s at positions 0,1,2,3,4 (for the 5 consecutive positions)
    # We want to check if bitset has 1s at positions s, s+1, s+2, s+3, s+4
    # This is equivalent to checking if (bitset >> s) & 0b11111 == 0b11111
    
    # Search for the smallest s in [1, p-5] such that s, s+1, s+2, s+3, s+4 are all non-zero quadratic residues
    # Note: s+i must be in [1, p-1], so s <= p-5 = 1004
    for s in range(1, p - 4):  # s from 1 to 1004
        if time.time() > deadline:
            break
        if (bitset >> s) & 0b11111 == 0b11111:
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
    
    raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found')
