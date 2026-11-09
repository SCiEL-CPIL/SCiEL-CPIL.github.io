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
    bitset = 0
    for v in range(1, p):
        if v in roots:
            bitset |= (1 << v)
    
    # We need to find the smallest s in [1, p-5] such that
    # s, s+1, s+2, s+3, s+4 are all non-zero quadratic residues.
    
    # M will have bit s set if B has bits s, s+1, s+2, s+3, s+4 set.
    # B >> k has bit s set if B has bit s+k set.
    # So M = B & (B >> 1) & (B >> 2) & (B >> 3) & (B >> 4)
    M = bitset & (bitset >> 1) & (bitset >> 2) & (bitset >> 3) & (bitset >> 4)
    
    # We require 1 <= s <= 1004 (which is p-5).
    # Create a mask for valid s values: bits 1 to 1004 set.
    # Bit i is set for i in [1, 1004].
    # (1 << 1005) - 1 gives bits 0..1004 set.
    # We want to clear bit 0.
    valid_mask = ((1 << 1005) - 1) & ~1
    
    M_valid = M & valid_mask
    
    if M_valid == 0:
        # Fallback: should not happen, but handle gracefully
        # Direct search
        best_s = None
        for s in range(1, p - 4):
            if all((s + i) in roots for i in range(5)):
                best_s = s
                break
        if best_s is None:
            raise RuntimeError('No run of 5 consecutive non-zero QRs found')
    else:
        # Find the least significant set bit in M_valid
        # This gives the smallest valid s
        lsb = M_valid & -M_valid
        best_s = lsb.bit_length() - 1
        
        # Double check just in case of any edge case logic errors
        if not (1 <= best_s <= 1004):
             # This shouldn't happen given the mask, but if it does, fallback
             best_s = None
             for s in range(1, p - 4):
                 if all((s + i) in roots for i in range(5)):
                     best_s = s
                     break
             if best_s is None:
                 raise RuntimeError('No run of 5 consecutive non-zero QRs found')
    
    result_roots = [roots[best_s + i] for i in range(5)]
    
    return {'start': best_s, 'roots': result_roots}
# EVOLVE-BLOCK-END
