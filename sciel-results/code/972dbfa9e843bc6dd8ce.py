"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    m = 5
    
    # Precompute quadratic residues and their roots
    # roots[r] will store one root x such that x*x % p == r
    roots = [0] * p
    mask = 0  # Bitset where bit i is 1 if i is a non-zero quadratic residue
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            # Set the bit for this residue
            mask |= (1 << r)
            # Store the first root found for this residue
            if roots[r] == 0:
                roots[r] = x
    
    # Search for the minimum start s
    # We need s, s+1, s+2, s+3, s+4 to all be quadratic residues
    # s ranges from 1 to 1004 (since s+4 <= 1008)
    for s in range(1, 1005):
        # Check if bits s through s+4 are set in mask
        # (mask >> s) shifts the bit at position s to position 0
        # We check the lowest 5 bits
        if ((mask >> s) & 0x1F) == 0x1F:
            # Found the start
            r_list = [roots[s + i] for i in range(m)]
            return {'start': s, 'roots': r_list}
    
    # If no run found (should not happen for p=1009), raise error
    raise RuntimeError('No run of 5 consecutive quadratic residues found')
# EVOLVE-BLOCK-END
