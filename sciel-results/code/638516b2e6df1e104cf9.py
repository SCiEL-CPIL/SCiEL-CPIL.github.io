import time

# EVOLVE-BLOCK-START
def construct():
    p = 1009
    m = 5
    
    # Precompute is_qr and roots
    # is_qr[r] = True if r is a non-zero QR, False otherwise
    # roots[r] = smallest x in [1, p-1] such that x^2 mod p == r
    is_qr = [False] * p
    roots = [0] * p
    
    for x in range(1, p):
        r = (x * x) % p
        if r == 0:
            continue
        if not is_qr[r]:
            is_qr[r] = True
            roots[r] = x
    
    # Search for minimum s
    # s ranges from 1 to p - m
    # We need s, s+1, ..., s+m-1 to be QRs
    # Note: s+i can be up to (p-m) + (m-1) = p-1. So no wraparound needed for indices.
    
    for s in range(1, p - m + 1):
        # Check if all s..s+m-1 are QRs
        # Using a loop is fine for m=5
        valid = True
        for i in range(m):
            if not is_qr[s + i]:
                valid = False
                break
        
        if valid:
            # Found the minimum s
            res_roots = [roots[s + i] for i in range(m)]
            return {'start': s, 'roots': res_roots}
            
    # If no run found (should not happen for p=1009, m=5)
    return {'start': p - m, 'roots': [0] * m} # Fallback, though score 0 is better for failure
# EVOLVE-BLOCK-END
