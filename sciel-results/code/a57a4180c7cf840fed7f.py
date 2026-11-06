"""Rewritten program using bitwise convolution for quadratic residue run detection."""

def construct():
    """
    Find the minimum start s (1 <= s <= 1004) such that s, s+1, ..., s+4
    are all non-zero quadratic residues modulo 1009, with explicit roots.
    
    Uses a bitset representation where bit i is set iff i is a non-zero QR.
    Then scans for the smallest s where 5 consecutive bits are all set.
    """
    p = 1009
    
    # Precompute quadratic residues as a bitset and store one root per residue
    # bit i set means i is a non-zero quadratic residue mod p
    qr_bits = 0  # integer bitset
    roots = {}   # residue -> one valid root (1 <= root <= p-1)
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            # Set bit r in the bitset
            qr_bits |= (1 << r)
            # Store the smallest root we see (for determinism)
            if r not in roots:
                roots[r] = x
    
    # Now find the smallest s in [1, 1004] such that bits s, s+1, s+2, s+3, s+4 are all set
    mask = 0b11111  # 5 consecutive bits must be 1
    
    for s in range(1, 1005):
        # Check if bits s through s+4 are all set
        if (qr_bits >> s) & mask == mask:
            result_roots = [roots[s + i] for i in range(5)]
            return {'start': s, 'roots': result_roots}
    
    # Should not reach here for p=1009, but just in case
    raise RuntimeError('No valid run of 5 consecutive non-zero QRs found')
