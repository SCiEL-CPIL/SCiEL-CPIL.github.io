"""Find minimum start s for 5 consecutive quadratic residues mod 1009."""
import time

def construct():
    p = 1009
    start_time = time.time()
    
    # Precompute roots: for each non-zero QR, store one root
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            if r not in roots:
                roots[r] = x
    
    # Create a boolean array: is_qr[i] = True if i is a non-zero QR
    is_qr = [False] * (p + 1)
    for r in roots:
        is_qr[r] = True
    
    # Build the bitset as a Python int. Bit i corresponds to residue i.
    bitset = 0
    for r in range(1, p):
        if is_qr[r]:
            bitset |= (1 << r)
    
    # Compute the AND of shifted versions to find all positions where
    # 5 consecutive residues are all QRs
    mask = bitset & (bitset >> 1) & (bitset >> 2) & (bitset >> 3) & (bitset >> 4)
    
    # Find the least s in [1, 1004] such that bit s is set in mask
    best_s = None
    for s in range(1, 1005):
        if (mask >> s) & 1:
            best_s = s
            break
    
    if best_s is None:
        # Fallback: linear scan
        for s in range(1, 1005):
            if all(is_qr[s + i] for i in range(5)):
                best_s = s
                break
    
    if best_s is None:
        raise RuntimeError('No valid run found')
    
    # Verify and extract roots
    result_roots = []
    for i in range(5):
        residue = best_s + i
        # Verify it's a QR
        assert is_qr[residue], f"Residue {residue} is not a QR"
        # Verify the root
        root = roots[residue]
        assert (root * root) % p == residue, f"Root verification failed for {residue}"
        result_roots.append(root)
    
    return {'start': best_s, 'roots': result_roots}
