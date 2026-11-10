"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Build a bitset of quadratic residues modulo p (excluding 0).
    # bit i is set if i is a nonzero quadratic residue mod p.
    # We need 5 consecutive residues starting at s, i.e., s, s+1, s+2, s+3, s+4 all nonzero QRs.
    # Use Python integers as arbitrary-precision bitsets.
    
    # Create bitset: bit i set means i is a quadratic residue (and i != 0)
    residues = 0
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            residues |= (1 << r)
    
    # We need to find the smallest s in [1, 1004] such that bits s, s+1, s+2, s+3, s+4 are all set.
    # This is equivalent to: (residues >> s) & 0x1F == 0x1F, but we need to be careful about wraparound.
    # Since s <= 1004 and we need s+4 <= 1008, and p=1009, indices 1..1008 are valid.
    # Bit 0 corresponds to residue 0, which we excluded.
    
    # For a given s, check if bits s through s+4 are all set:
    # mask = 0x1F << s
    # (residues & mask) == mask
    
    # We want the minimum s. Let's iterate s from 1 upward.
    # But we can do better: compute the AND of shifted versions.
    # Define: for offset i in 0..4, let shifted_i = residues >> i
    # Then s is valid iff (residues >> s) & 0x1F == 0x1F
    # Equivalently, for all i in 0..4, bit (s+i) in residues is set.
    
    # Alternative: compute a new bitset "runs" where bit s is set if s..s+4 are all QRs.
    # runs = residues & (residues >> 1) & (residues >> 2) & (residues >> 3) & (residues >> 4)
    # Then bit s in runs is set iff bits s, s+1, s+2, s+3, s+4 in residues are all set.
    # Wait: (residues >> i) has bit j set iff residues has bit j+i set.
    # So (residues & (residues>>1) & ... & (residues>>4)) has bit s set iff residues has bits s, s+1, ..., s+4 set.
    # Yes! That's exactly what we want.
    
    runs = residues
    for i in range(1, 5):
        runs &= (residues >> i)
    
    # Now find the smallest s >= 1 such that bit s is set in runs.
    # We need s <= 1004 (so that s+4 <= 1008 < p).
    # Iterate s from 1 to 1004.
    for s in range(1, 1005):
        if (runs >> s) & 1:
            # Found the start. Now find roots for s, s+1, ..., s+4.
            # We need roots such that root**2 % p == s+i.
            # We stored only one root per residue earlier, but let's recompute or store them.
            # Let's build a dict from residue to root.
            roots = {}
            for x in range(1, p):
                r = (x * x) % p
                if r != 0 and r not in roots:
                    roots[r] = x
            root_list = [roots[s + i] for i in range(5)]
            return {'start': s, 'roots': root_list}
    
    raise RuntimeError('No run of 5 consecutive quadratic residues found')
# EVOLVE-BLOCK-END
