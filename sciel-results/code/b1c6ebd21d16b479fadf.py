import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s wall-clock limit
    
    p = 1009
    m = 5
    
    # Build a bitset of quadratic residues modulo p
    # bit i is set if i is a nonzero quadratic residue mod p
    residue_bits = 0
    root_map = [0] * p
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            residue_bits |= (1 << r)
            if root_map[r] == 0:
                root_map[r] = x
    
    # Use bitwise AND of shifted bitsets to find smallest s where
    # s, s+1, s+2, s+3, s+4 are all quadratic residues
    # We need bits s, s+1, s+2, s+3, s+4 all set in residue_bits
    # Equivalently: (residue_bits >> s) & 0x1F == 0x1F
    
    # To find the smallest such s, we can AND shifted versions:
    # candidate = residue_bits & (residue_bits >> 1) & (residue_bits >> 2) & (residue_bits >> 3) & (residue_bits >> 4)
    # The lowest set bit in candidate gives the smallest s
    # But we need to be careful: bit s in candidate is set iff bits s..s+4 are all set in residue_bits
    
    candidate = residue_bits
    for shift in range(1, m):
        candidate &= (residue_bits >> shift)
    
    # Find the lowest set bit position >= 1 and <= 1004
    # We need 1 <= s <= 1004 (i.e., s+4 <= 1008 < p)
    # Mask out bit 0 (we need s >= 1) and bits > 1004
    mask = (1 << 1005) - 2  # bits 1 through 1004 set
    candidate &= mask
    
    if candidate != 0:
        # Find lowest set bit
        s = (candidate & (-candidate)).bit_length() - 1
        roots = [root_map[s + i] for i in range(m)]
        return {'start': s, 'roots': roots}
    
    # Fallback: sequential scan (should not happen for p=1009)
    for s in range(1, 1005):
        if all(root_map[s + i] != 0 for i in range(m)):
            roots = [root_map[s + i] for i in range(m)]
            return {'start': s, 'roots': roots}
    
    return {'start': 1, 'roots': [0, 0, 0, 0, 0]}
