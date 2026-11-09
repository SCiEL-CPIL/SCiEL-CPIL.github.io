"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    deadline = time.time() + 85.0  # safety margin before 90s limit
    
    # Precompute: for each quadratic residue value (1..p-1), store one valid root
    roots = {}
    for x in range(1, p):
        v = (x * x) % p
        if v != 0:  # exclude zero residues
            if v not in roots:
                roots[v] = x
    
    # Build a bitset where bit i is set if i is a non-zero quadratic residue
    # We need to find the smallest s (1 <= s <= p-5) such that
    # s, s+1, s+2, s+3, s+4 are all non-zero quadratic residues
    # i.e., bits s through s+4 are all set in the residue bitset
    
    # Create the bitset as an integer
    bitset = 0
    for v in range(1, p):
        if v in roots:
            bitset |= (1 << v)
    
    # We need to find the smallest s in [1, p-5] such that
    # (bitset >> s) & 0x1F == 0x1F  (bits s..s+4 are all set)
    # 
    # But we must also ensure s+4 < p (i.e., s <= p-5 = 1004)
    # and s >= 1
    
    # Mask for 5 consecutive bits
    mask5 = 0x1F  # 0b11111
    
    best_s = None
    for s in range(1, p - 4):  # s from 1 to 1004
        if (bitset >> s) & mask5 == mask5:
            best_s = s
            break
    
    if best_s is None:
        # Fallback: should not happen for p=1009, but handle gracefully
        # Try a direct search
        for s in range(1, p - 4):
            if all((s + i) in roots for i in range(5)):
                best_s = s
                break
    
    if best_s is None:
        raise RuntimeError('No run of 5 consecutive non-zero QRs found')
    
    result_roots = [roots[best_s + i] for i in range(5)]
    
    return {'start': best_s, 'roots': result_roots}
# EVOLVE-BLOCK-END
