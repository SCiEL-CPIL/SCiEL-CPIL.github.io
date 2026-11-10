"""Find the earliest run of 5 consecutive quadratic residues modulo 1009."""

def construct():
    p = 1009
    half = p // 2  # 504

    # Compute all quadratic residues and store one root per residue
    # r^2 mod p for r in 1..504 gives 504 distinct nonzero QRs
    # (since p ≡ 1 mod 4, no r in 1..504 has r^2 ≡ 0 mod p)
    root_map = {}
    for r in range(1, half + 1):
        val = (r * r) % p
        if val not in root_map:
            root_map[val] = r

    # Build a bitset where bit x is set iff x is a nonzero quadratic residue
    # bit 0 unused (zero excluded), bits 1..1008 represent values 1..1008
    mask = 0
    for v in root_map:
        mask |= (1 << v)

    # Find all positions s where s, s+1, s+2, s+3, s+4 are all QR
    # A bit at position s in the result means s is a valid start
    # We need: bit s, s+1, s+2, s+3, s+4 all set in mask
    # This is equivalent to: mask & (mask >> 1) & (mask >> 2) & (mask >> 3) & (mask >> 4)
    # has bit s set iff all five positions are QRs
    valid_starts = mask & (mask >> 1) & (mask >> 2) & (mask >> 3) & (mask >> 4)

    # Find the smallest s >= 1 with a valid start
    # We need s in [1, 1004] so that s+4 <= 1008
    # Extract the lowest set bit position
    # valid_starts has bit i set if i is a valid start (i.e., bits i..i+4 all QR)
    # We want the smallest i >= 1
    # Clear bit 0 if present (s=0 is not allowed)
    valid_starts &= ~1  # clear bit 0

    # Find the position of the lowest set bit
    if valid_starts == 0:
        raise RuntimeError('No run of 5 consecutive quadratic residues found')

    # Lowest set bit position: (valid_starts & -valid_starts).bit_length() - 1
    lowest = valid_starts & (-valid_starts)
    s = lowest.bit_length() - 1

    roots = [root_map[s + i] for i in range(5)]
    return {'start': s, 'roots': roots}
