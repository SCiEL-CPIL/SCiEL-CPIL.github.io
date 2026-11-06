"""
Find the minimal run-start s for 5 consecutive non-zero quadratic residues modulo 1009.
Uses a bitset approach for efficient determination of the minimal s.
"""
import time

def construct():
    p = 1009
    # Safety deadline guard
    start_time = time.time()
    deadline = start_time + 45
    
    # 1. Construct the bitset of non-zero quadratic residues
    # bit i is set if i is a non-zero QR mod p
    bits = 0
    is_res = [False] * p
    roots = [0] * p
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            # Set the bit for residue r
            bits |= (1 << r)
            is_res[r] = True
            # Store the smallest root (first encountered)
            if roots[r] == 0:
                roots[r] = x
        
        # Deadline check inside loop (unlikely to be needed for p=1009, but good practice)
        if time.time() > deadline:
            # Fallback: return a valid result if possible, or raise
            # Since we haven't finished building bits, we can't use the bitset method.
            # But p=1009 is tiny, this won't happen.
            raise RuntimeError("Time limit exceeded during residue calculation")

    # 2. Use bit operations to find all valid starting positions
    # We want s such that bits[s], bits[s+1], bits[s+2], bits[s+3], bits[s+4] are all 1.
    # This is equivalent to: (bits & (bits >> 1) & (bits >> 2) & (bits >> 3) & (bits >> 4)) has bit s set.
    
    valid_starts_bits = bits
    for i in range(1, 5):
        valid_starts_bits &= (bits >> i)
    
    # 3. Find the smallest s >= 1 such that bit s is set in valid_starts_bits
    # We iterate from 1 upwards.
    # Since p is small, a simple loop is fine and clear.
    # Alternatively, use bit tricks to find lowest set bit.
    
    min_s = -1
    # Mask out bit 0, as s must be >= 1
    # Also, s+4 <= p-1 => s <= p-5 = 1004.
    # The bits for s > 1004 might be set if residues exist, but we are constrained by the problem statement "1<=s<=1004".
    # Actually, if s=1005, s+4=1009=0 mod 1009. 0 is excluded.
    # So we only care about s in [1, 1004].
    
    for s in range(1, p - 4):
        if time.time() > deadline:
            raise RuntimeError("Time limit exceeded during search")
            
        # Check if bit s is set in valid_starts_bits
        if (valid_starts_bits >> s) & 1:
            min_s = s
            break
            
    if min_s == -1:
        raise RuntimeError("No run of 5 consecutive quadratic residues found")
        
    # 4. Construct the roots
    roots_list = [roots[min_s + i] for i in range(5)]
    
    return {'start': min_s, 'roots': roots_list}
