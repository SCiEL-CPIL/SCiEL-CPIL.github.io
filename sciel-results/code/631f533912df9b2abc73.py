"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    deadline = time.time() + 60.0
    
    # Precompute: for each residue r in [1, p-1], store one root x in [1, p-1]
    # such that x*x % p == r. We want to minimize s, so we just need any valid root.
    roots = {}
    # Also build a bitset for fast consecutive QR checks
    # Bit i is 1 if i is a non-zero QR, 0 otherwise.
    qr_mask = 0
    
    for x in range(1, p):
        v = (x * x) % p
        if v == 0:
            continue  # zero residues excluded
        if v not in roots:
            roots[v] = x
            qr_mask |= (1 << v)
    
    # Forward search: find the smallest s in [1, p-5] such that
    # s, s+1, s+2, s+3, s+4 are all quadratic residues (non-zero).
    # Note: s+i must be in [1, p-1] for all i in [0,4], so s <= p-5 = 1004.
    # We use bitwise operations to check if 5 consecutive bits starting at s are set.
    # Mask for 5 bits: 0b11111 = 31
    mask5 = 31
    
    for s in range(1, p - 4):  # s from 1 to 1004
        # Check if bits s, s+1, s+2, s+3, s+4 are all set in qr_mask
        # (qr_mask >> s) & mask5 checks exactly these 5 bits
        if ((qr_mask >> s) & mask5) == mask5:
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
    
    # Fallback: descending search (should not be reached)
    for s in range(p - 5, 0, -1):
        if ((qr_mask >> s) & mask5) == mask5:
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
    
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
