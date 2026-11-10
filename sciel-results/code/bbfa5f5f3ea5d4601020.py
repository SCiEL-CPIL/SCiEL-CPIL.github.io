"""Find the minimum s such that s, s+1, ..., s+4 are all non-zero quadratic residues mod 1009.
Optimal solution found at s=1.
"""

import time

def construct():
    p = 1009
    m = 5
    deadline = time.monotonic() + 60.0
    
    # Precompute quadratic residues using a bitset for fast lookup
    # is_qr[v] is True if v is a non-zero quadratic residue mod p
    is_qr = [False] * p
    roots = [0] * p
    
    for x in range(1, p):
        r2 = (x * x) % p
        if not is_qr[r2]:
            is_qr[r2] = True
            roots[r2] = x
    
    # Build a bitset where bit i is set if i is a QR (for i in 1..p-1)
    # We'll use Python integers as arbitrary-precision bitsets
    qr_bits = 0
    for v in range(1, p):
        if is_qr[v]:
            qr_bits |= (1 << v)
    
    # Use bitwise AND of shifted bitsets to find all valid starts simultaneously
    # matches has bit s set iff s, s+1, ..., s+m-1 are all QRs
    matches = qr_bits
    for i in range(1, m):
        matches &= (qr_bits >> i)
        
    # We require s >= 1, so clear bit 0
    matches &= ~1
    
    if matches == 0:
        raise RuntimeError('No run found')
        
    # Find the least significant set bit
    # lsb = matches & -matches
    # s = lsb.bit_length() - 1
    lsb = matches & -matches
    s = lsb.bit_length() - 1
    
    # Verify s is within bounds (1 <= s <= p-m)
    if s < 1 or s > p - m:
        raise RuntimeError('No valid run in range')
        
    return {'start': s, 'roots': [roots[s + i] for i in range(m)]}
