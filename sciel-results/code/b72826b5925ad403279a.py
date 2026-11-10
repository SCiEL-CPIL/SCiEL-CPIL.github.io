"""Rewritten program: find the minimum start s for 5 consecutive quadratic residues mod 1009."""
import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s wall-clock kill
    p = 1009
    
    # Precompute for each residue r in 1..p-1 whether it's a quadratic residue
    # and store one root for it
    # A number r (1<=r<=p-1) is a QR mod p iff there exists x in 1..p-1 with x^2 ≡ r (mod p)
    
    # Build a boolean array is_qr[r] = True if r is a QR mod p (r in 1..p-1)
    # and root[r] = one x such that x^2 % p == r
    is_qr = [False] * p  # index 0..p-1, we only use 1..p-1
    root = [0] * p
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:  # exclude zero residue
            if not is_qr[r]:
                is_qr[r] = True
                root[r] = x
    
    # Now search for the minimum s in 1..1004 such that s, s+1, s+2, s+3, s+4 are all QRs
    # Note: s+i must be in range 1..1008 (since residues are 1..p-1=1008)
    # So s can be at most 1004 (so s+4=1008)
    # And s >= 1
    
    # For each s, check if all of s, s+1, s+2, s+3, s+4 are QRs
    # We need s+i <= 1008 for all i in 0..4, so s <= 1004
    # And s >= 1
    
    best_s = None
    best_roots = None
    
    for s in range(1, 1005):
        if time.time() > deadline:
            break
        # Check if s, s+1, s+2, s+3, s+4 are all QRs
        ok = True
        for i in range(5):
            r = s + i
            if r < 1 or r > 1008 or not is_qr[r]:
                ok = False
                break
        if ok:
            best_s = s
            best_roots = [root[s + i] for i in range(5)]
            break  # we want the minimum s, so first one found is optimal
    
    if best_s is None:
        # Fallback: should not happen, but just in case
        # Try a broader search or raise
        raise RuntimeError('No run of 5 consecutive QRs found')
    
    return {'start': best_s, 'roots': best_roots}
