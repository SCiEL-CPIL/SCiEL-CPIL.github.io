"""Rewritten program using bitwise convolution approach for finding earliest run of 5 consecutive quadratic residues."""

def construct():
    p = 1009
    # Build bitset: bit i is set if i is a non-zero quadratic residue mod p
    # We need residues in range 1..1008 (excluding 0)
    residue_set = set()
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            residue_set.add(r)
    
    # Create bitset integer where bit i represents whether i is a residue
    # Bit 0 is unused (or represents 0), bits 1..1008 represent residues 1..1008
    bitset = 0
    for r in residue_set:
        bitset |= (1 << r)
    
    # Find smallest s (1 <= s <= 1004) such that s, s+1, s+2, s+3, s+4 are all residues
    # Check: (bitset >> s) & 0b11111 == 0b11111
    mask = 0b11111  # 5 consecutive bits
    for s in range(1, 1005):
        if (bitset >> s) & mask == mask:
            # Found the start, now find roots for each residue
            roots = []
            for i in range(5):
                val = s + i
                # Find a root x such that x^2 mod p == val, with 1 <= x <= 1008
                # We can search for it
                root = None
                for x in range(1, p):
                    if (x * x) % p == val:
                        root = x
                        break
                roots.append(root)
            return {'start': s, 'roots': roots}
    
    raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found')
