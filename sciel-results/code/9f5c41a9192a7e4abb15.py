"""Rewritten program to find the minimum start of a run of 5 quadratic residues mod 1009."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    deadline = time.time() + 80.0
    
    # Map each quadratic residue to its smallest positive root in [1, p-1]
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r not in roots:
            roots[r] = x
    
    # Build a bitset where bit i is 1 if (i+1) is a quadratic residue
    # Bit position i corresponds to residue value i+1
    bitset = 0
    for r in roots:
        bitset |= (1 << (r - 1))
    
    # To find 5 consecutive residues starting at s:
    # We need bits (s-1), s, (s+1), (s+2), (s+3) all set in bitset
    # This is equivalent to: (bitset >> (s-1)) & 0x1F == 0x1F
    # We want the minimum s >= 1 such that this holds
    
    # Use bit manipulation to find the leftmost (smallest) position where 5 consecutive bits are set
    # Shift bitset right by 0, 1, 2, 3, 4 and AND them together
    # The resulting bitset has bit i set iff bits i, i+1, i+2, i+3, i+4 are all set in original
    # But we want consecutive starting at s, so bit (s-1) in result means s..s+4 are residues
    
    # AND of shifted versions:
    # result_bit[i] = bitset[i] & bitset[i+1] & bitset[i+2] & bitset[i+3] & bitset[i+4]
    # So: result = bitset & (bitset >> 1) & (bitset >> 2) & (bitset >> 3) & (bitset >> 4)
    
    result = bitset & (bitset >> 1) & (bitset >> 2) & (bitset >> 3) & (bitset >> 4)
    
    # Find the least significant set bit in result
    # This gives us the smallest i such that residues i+1, i+2, i+3, i+4, i+5 are all QRs
    # So s = i + 1
    
    if result == 0:
        # Fallback to linear search
        for s in range(1, 1005):
            if time.time() > deadline:
                break
            if all((s + i) in roots for i in range(5)):
                res_roots = [roots[s + i] for i in range(5)]
                return {'start': s, 'roots': res_roots}
        raise RuntimeError("No run of 5 quadratic residues found")
    
    # Find least set bit position
    # result & (-result) isolates the least significant set bit
    least_bit = result & (-result)
    # Get the index of that bit
    i = least_bit.bit_length() - 1
    s = i + 1
    
    # Verify s is in valid range
    if s < 1 or s > 1004:
        # Fallback
        for s in range(1, 1005):
            if time.time() > deadline:
                break
            if all((s + i) in roots for i in range(5)):
                res_roots = [roots[s + i] for i in range(5)]
                return {'start': s, 'roots': res_roots}
        raise RuntimeError("No run of 5 quadratic residues found")
    
    res_roots = [roots[s + i] for i in range(5)]
    return {'start': s, 'roots': res_roots}
# EVOLVE-BLOCK-END
