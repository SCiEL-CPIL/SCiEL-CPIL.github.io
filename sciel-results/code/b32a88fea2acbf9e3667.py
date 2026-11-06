def construct():
    p = 1009
    # is_qr[i] will be True if i is a non-zero quadratic residue mod p
    is_qr = [False] * p
    # roots[r] will store a root x such that x^2 % p == r
    roots = {}

    # Precompute quadratic residues
    for x in range(1, p):
        r = (x * x) % p
        if not is_qr[r]:
            is_qr[r] = True
            roots[r] = x

    # Search for the minimum s
    # s must be such that s, s+1, s+2, s+3, s+4 are all in [1, p-1]
    # and are all quadratic residues.
    # Max s is p - 5 = 1004.
    for s in range(1, p - 4):
        if (is_qr[s] and is_qr[s+1] and is_qr[s+2] and 
            is_qr[s+3] and is_qr[s+4]):
            return {
                'start': s,
                'roots': [roots[s], roots[s+1], roots[s+2], roots[s+3], roots[s+4]]
            }
            
    # This should never happen for p=1009, but just in case
    raise RuntimeError("No valid start found")
