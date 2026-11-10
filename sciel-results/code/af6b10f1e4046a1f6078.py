import time

def construct():
    p = 1009
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Precompute QR bitset and roots
    # Bit i is 1 if i is a QR
    qr_bits = 0
    roots = {}
    
    for x in range(1, p):
        r = (x * x) % p
        # r is in 1..1008
        qr_bits |= (1 << r)
        # Store a root. Any root works. x is in 1..1008.
        if r not in roots:
            roots[r] = x
            
    # Find minimal s such that s, s+1, s+2, s+3, s+4 are all QRs
    # We check bits s..s+4.
    # mask = 0b11111 (5 bits)
    mask = 0b11111
    
    for s in range(1, p - 4):
        if time.time() > deadline:
            # This shouldn't happen for p=1009, but required by directive
            raise TimeoutError("Search exceeded deadline")
        
        # Check if bits s to s+4 are all set in qr_bits
        # Shift qr_bits right by s, then check lower 5 bits
        if (qr_bits >> s) & mask == mask:
            # Found the minimal s
            res_roots = [roots[s + i] for i in range(5)]
            return {'start': s, 'roots': res_roots}
            
    raise RuntimeError("No run found")
