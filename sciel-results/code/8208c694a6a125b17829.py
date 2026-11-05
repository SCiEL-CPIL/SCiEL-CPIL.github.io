def construct():
    p = 1009
    # is_qr[i] will be True if i is a quadratic residue mod p
    is_qr = [False] * p
    # roots[i] will store one valid root for i if it is a QR
    roots = [0] * p
    
    for x in range(1, p):
        r = (x * x) % p
        if not is_qr[r]:
            is_qr[r] = True
            roots[r] = x
            
    # Search for smallest s
    for s in range(1, p - 4):
        # Check if s, s+1, s+2, s+3, s+4 are all QRs
        if is_qr[s] and is_qr[s+1] and is_qr[s+2] and is_qr[s+3] and is_qr[s+4]:
            return {'start': s, 'roots': [roots[s], roots[s+1], roots[s+2], roots[s+3], roots[s+4]]}
            
    raise RuntimeError("No run found")
