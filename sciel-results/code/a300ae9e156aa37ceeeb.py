"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 60.0  # safety margin before 90s kill
    p = 1009
    m = 5  # run length
    k = 2  # exponent (quadratic)
    
    # Precompute quadratic residues and one root for each
    roots = {}
    for x in range(1, p):
        v = (x * x) % p
        if v not in roots:
            roots[v] = x
    
    # Build bitset of non-zero quadratic residues
    # bit i is 1 if i is a non-zero QR mod p
    bitset = 0
    for x in range(1, p):
        v = (x * x) % p
        bitset |= (1 << v)
    
    # Clear bit 0 (zero residue excluded)
    bitset &= ~(1 << 0)
    
    # Create a mask for m consecutive bits
    # We need to find the smallest s in [1, p-m] such that
    # bits s, s+1, ..., s+m-1 are all set in bitset
    
    # Method: for each shift i in [0, m-1], shift bitset right by i,
    # then AND all shifted versions. The resulting bitset has bit s set
    # iff s+i is set in the original for all i in [0, m-1].
    # Then find the lowest set bit >= 1 and <= p-m.
    
    window_mask = (1 << m) - 1  # m ones
    
    # Alternative approach: scan s from 1 upward
    # For each s, check if (bitset >> s) & window_mask == window_mask
    # This is O(p) checks, each O(1) with bitwise ops
    
    best_s = None
    for s in range(1, p - m + 1):
        if (bitset >> s) & window_mask == window_mask:
            best_s = s
            break
    
    if best_s is None:
        raise RuntimeError('No run of length 5 found')
    
    # Verify we're within deadline
    if time.time() > deadline:
        # Should not happen for this small problem, but guard anyway
        pass
    
    # Extract roots for s, s+1, ..., s+m-1
    result_roots = []
    for i in range(m):
        val = best_s + i
        r = roots[val]
        result_roots.append(r)
    
    return {'start': best_s, 'roots': result_roots}
# EVOLVE-BLOCK-END
