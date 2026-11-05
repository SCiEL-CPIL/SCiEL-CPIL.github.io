# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Bitmask where bit i is 1 if i is a non-zero quadratic residue mod p
    mask = 0
    roots = {}
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            mask |= (1 << r)
            if r not in roots:
                roots[r] = x
                
    # Find the smallest s such that s, s+1, s+2, s+3, s+4 are all residues
    # s must be in 1..1004 to keep s+4 <= 1008
    for s in range(1, 1005):
        if (mask >> s) & 0x1F == 0x1F:
            return {'start': s, 'roots': [roots[s+i] for i in range(5)]}
            
    raise RuntimeError('No run of 5 consecutive quadratic residues found')
# EVOLVE-BLOCK-END
