"""Find the minimum s such that s, s+1, ..., s+4 are all non-zero quadratic residues mod 1009."""

# EVOLVE-BLOCK-START
def construct():
    p = 1009
    m = 5

    # Precompute roots and build a bitset of non-zero quadratic residues.
    root_of = [0] * p  # root_of[v] = some r with r^2 % p == v, 0 if not a residue
    mask = 0  # bit v is set iff v is a non-zero QR mod p

    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            if root_of[r] == 0:
                root_of[r] = x
            mask |= (1 << r)

    # Use bitwise AND of shifted masks to find all s where s..s+m-1 are all residues.
    # For m=5: valid_s_mask = mask & (mask>>1) & (mask>>2) & (mask>>3) & (mask>>4)
    # Bit s is set iff bits s, s+1, s+2, s+3, s+4 are all set in mask.
    # We need 1 <= s <= p-1-m+1 = 1004, and s+4 <= 1008 = p-1.
    # Since mask has no bit 0 set (we excluded r==0), and we only care about s>=1,
    # the lowest set bit in valid_s_mask (with s>=1) gives the answer.

    shifted = mask
    for i in range(1, m):
        shifted &= (mask >> i)

    # Mask out bit 0 (s=0 is not allowed) and bits above p-1-m+1=1004
    # Actually, since mask has 0 at position 0, bit 0 of shifted is already 0.
    # But let's be safe: clear bit 0.
    shifted &= ~1

    # Find the lowest set bit position (that's the smallest s >= 1)
    # lowest set bit: shifted & (-shifted) gives a power of 2
    if shifted == 0:
        raise RuntimeError('No run of 5 consecutive non-zero QRs found')

    lowbit = shifted & (-shifted)
    s = lowbit.bit_length() - 1  # position of the lowest set bit

    # Verify s is in valid range [1, 1004]
    if s < 1 or s > p - m:
        # Fallback: scan upward
        for s2 in range(1, p - m + 1):
            ok = True
            for i in range(m):
                if not ((mask >> (s2 + i)) & 1):
                    ok = False
                    break
            if ok:
                s = s2
                break
        else:
            raise RuntimeError('No run of 5 consecutive non-zero QRs found')

    roots = [root_of[s + i] for i in range(m)]
    return {'start': s, 'roots': roots}
# EVOLVE-BLOCK-END
