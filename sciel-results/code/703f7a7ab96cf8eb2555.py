"""Find the minimum start s for 5 consecutive non-zero quadratic residues mod 1009."""

import time
import math

def construct():
    p = 1009
    deadline = time.time() + 85.0

    # Precompute: for each x in [1, p-1], compute x*x mod p.
    # Store the smallest root for each non-zero residue.
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            if r not in roots:
                roots[r] = x

    # Build an integer bitset: bit i is 1 iff i is a non-zero QR mod p.
    mask = 0
    for r in roots:
        mask |= (1 << r)

    # We need 5 consecutive bits set: check (mask >> s) & 0b11111 == 0b11111
    # Search ascending from s=1 to s=p-5=1004 to find the minimum valid start.
    target = 0b11111
    shifted = mask  # will shift right each iteration for efficiency

    # Use a direct loop with the bitset check
    # To avoid repeated shifting of a large integer, we can precompute
    # but the loop is at most 1004 iterations, so it's fast.
    
    for s in range(1, p - 4):
        if time.time() > deadline:
            break
        if (mask >> s) & target == target:
            r0, r1, r2, r3, r4 = s, s + 1, s + 2, s + 3, s + 4
            return {'start': s, 'roots': [roots[r0], roots[r1], roots[r2], roots[r3], roots[r4]]}

    raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found mod 1009')
