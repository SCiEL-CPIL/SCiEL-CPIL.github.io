def construct():
    p = 1009
    # Determine which numbers are quadratic residues
    is_qr = [False] * p
    for x in range(1, p):
        r = (x * x) % p
        is_qr[r] = True
    
    # Find the smallest s such that s, s+1, s+2, s+3, s+4 are all QRs
    for s in range(1, 1005):
        if all(is_qr[s+i] for i in range(5)):
            # Find roots for s..s+4
            roots = []
            for i in range(5):
                val = s + i
                # Find a root for val
                for x in range(1, p):
                    if (x * x) % p == val:
                        roots.append(x)
                        break
            return {'start': s, 'roots': roots}
    
    raise RuntimeError('No valid run found')
