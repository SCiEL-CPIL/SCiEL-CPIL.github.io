import time

def construct():
    deadline = time.time() + 85.0
    p = 1009
    m = 5
    
    # Precompute: for each non-zero residue r in [1, p-1], store a root x in [1, p-1]
    # such that x^2 % p == r. Keep the smallest x for each residue.
    root_of = [0] * (p + 1)
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and root_of[r] == 0:
            root_of[r] = x
    
    # Build bitset of non-zero quadratic residues
    res_mask = 0
    for r in range(1, p):
        if root_of[r] != 0:
            res_mask |= (1 << r)
    
    # Find minimum s such that s, s+1, ..., s+4 are all non-zero QRs
    # M has bit s set iff s..s+4 are all QRs
    M = res_mask
    for i in range(1, m):
        M &= (res_mask >> i)
    
    if M == 0:
        # Fallback: linear scan
        s = -1
        for candidate_s in range(1, p - m + 1):
            if all(root_of[candidate_s + i] != 0 for i in range(m)):
                s = candidate_s
                break
        if s == -1:
            raise RuntimeError("No solution found")
    else:
        # Find lowest set bit >= 1
        lowest_bit = M & -M
        s = lowest_bit.bit_length() - 1
        if s < 1:
            # Bit 0 might be set (0 is not a QR, so this shouldn't happen),
            # but just in case, find next set bit
            M2 = M & ~1  # clear bit 0
            if M2 == 0:
                raise RuntimeError("No valid s >= 1 found")
            lowest_bit = M2 & -M2
            s = lowest_bit.bit_length() - 1
    
    if s > p - m:
        raise RuntimeError(f"s={s} out of bounds")
    
    # Collect roots
    roots = []
    for i in range(m):
        val = s + i
        r = root_of[val]
        if r == 0:
            raise RuntimeError(f"No root for {val}")
        roots.append(r)
    
    return {'start': s, 'roots': roots}
