"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Build a bitset of quadratic residues mod p (excluding 0)
    # Bit i is 1 if i is a quadratic residue mod p (and i != 0)
    residue_bits = 0
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            residue_bits |= (1 << r)
    
    # We need to find the smallest s (1 <= s <= 1004) such that
    # s, s+1, s+2, s+3, s+4 are all quadratic residues mod p (non-zero).
    # 
    # Using bitsets: for a given s, we need bits s..s+4 all set.
    # Create a mask for positions 0..4: mask5 = (1<<5) - 1 = 0b11111
    # Check if (residue_bits >> s) & 0b11111 == 0b11111
    #
    # To find the minimum s efficiently, we can scan from s=1 upward.
    # But we can do better: compute which s values work by checking
    # (residue_bits & (residue_bits >> 1) & (residue_bits >> 2) & (residue_bits >> 3) & (residue_bits >> 4))
    # has bit s set.
    
    # combined_bits[s] = 1 iff s, s+1, s+2, s+3, s+4 are all residues
    combined = residue_bits & (residue_bits >> 1) & (residue_bits >> 2) & (residue_bits >> 3) & (residue_bits >> 4)
    
    # Now find the smallest s in [1, 1004] such that bit s is set in combined
    # and also ensure that s+4 <= 1008 (i.e., s <= 1004), which is already our range.
    # Also need to make sure residues are non-zero: since we excluded 0 from residue_bits,
    # if s=0 it would be excluded, but our range starts at 1.
    # However, we must also ensure that s+i != 0 for all i, which is true since s>=1 and p=1009.
    
    # Find the least set bit in combined that is in range [1, 1004]
    # Mask out bits 0 and bits > 1004
    valid_mask = 0
    for s in range(1, 1005):
        valid_mask |= (1 << s)
    
    candidates = combined & valid_mask
    if candidates == 0:
        raise RuntimeError('No valid run found')
    
    # Find the least significant set bit
    # candidates & (-candidates) gives the lowest set bit
    lowest = candidates & (-candidates)
    s = lowest.bit_length() - 1
    
    # Now find roots for s, s+1, ..., s+4
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and r not in roots:
            roots[r] = x
    
    root_list = [roots[s + i] for i in range(5)]
    
    return {'start': s, 'roots': root_list}
# EVOLVE-BLOCK-END
