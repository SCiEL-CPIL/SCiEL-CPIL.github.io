import time

def construct():
    p = 1009
    # Bitmask where bit i is 1 if i is a non-zero quadratic residue mod p
    mask = 0
    roots = {}
    
    # Precompute residues
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            mask |= (1 << r)
            if r not in roots:
                roots[r] = x
                
    # Search for minimum s
    # We need s, s+1, s+2, s+3, s+4 to be set in mask
    # This is equivalent to ((mask >> s) & 0x1F) == 0x1F
    for s in range(1, p - 4):
        if ((mask >> s) & 0x1F) == 0x1F:
            return {
                'start': s,
                'roots': [roots[s+i] for i in range(5)]
            }
            
    # Fallback if not found (should not happen for p=1009)
    raise RuntimeError("No run found")
