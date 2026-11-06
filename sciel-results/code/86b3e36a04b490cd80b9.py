import time

def construct():
    p = 1009
    deadline = time.time() + 85.0
    
    # Precompute roots for all non-zero quadratic residues
    # roots[r] = x such that x^2 % p == r and 1 <= x <= p-1
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            if r not in roots:
                roots[r] = x
    
    # Use a bitset representation for efficient consecutive-residue checking
    # bit i is 1 if i is a non-zero quadratic residue (1-indexed positions)
    # We'll use a Python integer where bit i (0-indexed) corresponds to value i+1
    # So value v has bit (v-1) set if v is a residue
    
    bitset = 0
    for r in roots:
        bitset |= (1 << (r - 1))
    
    # To find 5 consecutive residues starting at s:
    # We need bits (s-1), s, (s+1), (s+2), (s+3) all set
    # That's a mask of 5 consecutive 1s starting at position (s-1)
    # mask_5 = 0b11111 = 31
    # For start s, we check: (bitset >> (s-1)) & 31 == 31
    
    # But we need s in [1, 1004] so that s+4 <= 1008
    # s-1 ranges from 0 to 1003
    
    # Find the minimum s such that the 5-bit window starting at position s-1 is all 1s
    # We can scan from s=1 upward
    
    for s in range(1, 1005):
        if time.time() > deadline:
            # Fallback: return best found so far or raise
            # Since we're scanning in order, if we haven't found one yet, something is wrong
            # But for safety, let's just continue or break
            pass
        
        # Check if bits (s-1) through (s+3) are all set
        if (bitset >> (s - 1)) & 0x1F == 0x1F:
            r0 = roots[s]
            r1 = roots[s + 1]
            r2 = roots[s + 2]
            r3 = roots[s + 3]
            r4 = roots[s + 4]
            return {'start': s, 'roots': [r0, r1, r2, r3, r4]}
    
    raise RuntimeError("No valid run found")
