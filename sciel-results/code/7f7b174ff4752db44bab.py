import time

def construct():
    p = 1009
    m = 5
    deadline = time.time() + 80.0

    # Build a bitset of which values 1..1008 are quadratic residues,
    # and a mapping from residue to its smallest root in [1, p-1].
    # We use an integer as a bitset for fast "are all 5 consecutive
    # values residues?" checks via bit operations.

    # roots_map[r] = smallest x in [1, 1008] with x^2 % p == r
    roots_map = {}
    for x in range(1, p):
        r = (x * x) % p
        if r not in roots_map:
            roots_map[r] = x

    # Build bitset: bit i (0-indexed) is 1 if (i+1) is a quadratic residue
    # We index so that value v (1..1008) corresponds to bit (v-1)
    bitset = 0
    for r in roots_map:
        bitset |= (1 << (r - 1))

    # For a candidate start s (1-indexed), the 5 values s, s+1, ..., s+4
    # correspond to bits (s-1), s, s+1, s+2, s+3 in the bitset.
    # We need all 5 bits set. Mask for 5 consecutive bits starting at position (s-1):
    # mask = 0b11111 << (s-1)
    # Check: (bitset & mask) == mask

    # Precompute the 5-bit mask
    five_bit_mask = 0b11111

    # Iterate s from 1 to 1004
    for s in range(1, 1005):
        if time.time() > deadline:
            break
        shift = s - 1
        if (bitset & (five_bit_mask << shift)) == (five_bit_mask << shift):
            roots = []
            for i in range(m):
                val = s + i
                roots.append(roots_map[val])
            return {'start': s, 'roots': roots}

    raise RuntimeError('No run found')
