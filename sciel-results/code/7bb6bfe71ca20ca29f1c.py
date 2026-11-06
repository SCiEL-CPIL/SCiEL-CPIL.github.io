"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Build a bitset of quadratic residues mod p (excluding 0)
    # Bit i is 1 if i is a quadratic residue mod p (and i != 0)
    residue_bits = 0
    # Precompute roots for O(1) lookup
    roots = [0] * p
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            residue_bits |= (1 << r)
            # Store the first (smallest) root found for each residue
            if roots[r] == 0:
                roots[r] = x
    
    # We need to find the smallest s (1 <= s <= 1004) such that
    # s, s+1, s+2, s+3, s+4 are all quadratic residues mod p (non-zero).
    
    # combined_bits[s] = 1 iff s, s+1, s+2, s+3, s+4 are all residues
    # We perform bitwise AND of shifted versions of residue_bits
    combined = residue_bits & (residue_bits >> 1) & (residue_bits >> 2) & (residue_bits >> 3) & (residue_bits >> 4)
    
    # The lowest set bit in 'combined' corresponds to the smallest s
    # where the condition holds.
    # Since 0 is not a quadratic residue (excluded), bit 0 of residue_bits is 0,
    # so bit 0 of combined is 0.
    # For s > 1004, the range s..s+4 extends beyond 1008.
    # residue_bits has 0s for indices >= 1009.
    # So for s=1005, we check bit 1005 in residue_bits (0), so combined bit 1005 is 0.
    # Thus, any set bit in combined is a valid s in [1, 1004].
    
    if combined == 0:
        raise RuntimeError('No valid run found')
    
    # Find the least significant set bit
    lowest_bit = combined & (-combined)
    s = lowest_bit.bit_length() - 1
    
    # Extract roots for s, s+1, ..., s+4
    root_list = [roots[s + i] for i in range(5)]
    
    return {'start': s, 'roots': root_list}
# EVOLVE-BLOCK-END
