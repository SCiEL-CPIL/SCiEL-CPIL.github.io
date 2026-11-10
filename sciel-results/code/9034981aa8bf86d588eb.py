"""Rewritten program using a direct, optimized search for the minimum run-start."""
import time

# EVOLVE-BLOCK-START
def construct():
    p = 1009
    m = 5
    
    # Deadline guard
    start_time = time.time()
    deadline = 85.0  # 5s margin
    
    # Precompute the smallest root for each non-zero quadratic residue modulo p.
    # Use a list for O(1) access without hash overhead.
    # root_of[r] = smallest x such that x*x % p == r, or 0 if r is not a residue.
    root_of = [0] * p
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and root_of[r] == 0:
            root_of[r] = x

    # Build a bitset where bit i is set if i is a non-zero quadratic residue.
    # We index 0..p-1, but only care about 1..p-1 (0 is excluded).
    # bit i = 1 if root_of[i] != 0
    bitset = 0
    for i in range(1, p):
        if root_of[i] != 0:
            bitset |= (1 << i)

    # Use bitwise AND to find the first run of m consecutive residues.
    # mask5 will have bit s set if bits s, s+1, ..., s+m-1 are all set in bitset.
    mask5 = bitset
    for k in range(1, m):
        mask5 &= (bitset >> k)
        
    # Find the lowest set bit in mask5
    if mask5 == 0:
        raise RuntimeError('No run found')
        
    # The position of the lowest set bit is the start index s.
    # (mask5 & -mask5) isolates the lowest set bit.
    # bit_length() - 1 gives the index.
    s = (mask5 & -mask5).bit_length() - 1
    
    # Verify s is within bounds (1 <= s <= p-m)
    if s < 1 or s > p - m:
        raise RuntimeError('Run out of bounds')
        
    return {'start': s, 'roots': [root_of[s+i] for i in range(m)]}
# EVOLVE-BLOCK-END
