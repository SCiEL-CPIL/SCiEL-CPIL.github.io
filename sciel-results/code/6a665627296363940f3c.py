# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Build bitset of non-zero quadratic residues.
    # bit i is 1 iff i is a QR mod p.
    bitset = 0
    # Also prepare roots lookup.
    # We need a root for each residue.
    # roots[r] = x such that x^2 % p == r.
    # We want x in [1, p-1].
    roots = [0] * p  # Index 0..1008
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            bitset |= (1 << r)
            # Store the first root found (smallest x)
            if roots[r] == 0:
                roots[r] = x
    
    # Find the smallest s in [1, 1004] such that s..s+4 are all QRs.
    # consec bit s is 1 iff bits s, s+1, s+2, s+3, s+4 are set in bitset.
    consec = bitset & (bitset >> 1) & (bitset >> 2) & (bitset >> 3) & (bitset >> 4)
    
    # Scan for the first set bit in consec, starting from s=1.
    # We can use bit_length or scanning.
    # Since we want the *lowest* index, we scan from 1 upwards.
    for s in range(1, 1005):
        if (consec >> s) & 1:
            # Found s.
            # Extract roots for s, s+1, s+2, s+3, s+4.
            res_roots = [roots[s + i] for i in range(5)]
            return {'start': s, 'roots': res_roots}
    
    raise RuntimeError('No run of five consecutive non-zero quadratic residues found')
# EVOLVE-BLOCK-END
