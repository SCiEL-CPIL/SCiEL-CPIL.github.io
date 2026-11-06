"""Bitwise approach for finding earliest run of 5 consecutive quadratic residues mod 1009."""

def construct():
    p = 1009
    
    # Build bitset: bit i is set if i is a non-zero quadratic residue mod p
    bitset = 0
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            bitset |= (1 << r)
    
    # Find smallest s (1 <= s <= 1004) such that s, s+1, s+2, s+3, s+4 are all residues
    mask = 0b11111  # 5 consecutive bits
    found_s = None
    for s in range(1, 1005):
        if (bitset >> s) & mask == mask:
            found_s = s
            break
    
    if found_s is None:
        raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found')
    
    # Precompute roots: for each residue value, store one root x (1 <= x <= 1008)
    roots_map = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and r not in roots_map:
            roots_map[r] = x
    
    # Build the roots list
    roots = []
    for i in range(5):
        val = found_s + i
        roots.append(roots_map[val])
    
    return {'start': found_s, 'roots': roots}
