import time

def construct():
    deadline = time.time() + 85  # 5s safety margin before 90s kill
    p = 1009
    
    # Precompute quadratic residues and their roots
    # is_qr[r] is True if r is a non-zero quadratic residue mod p
    # roots[r] stores a valid root x such that x*x % p == r
    is_qr = [False] * p
    roots = [0] * p
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            if not is_qr[r]:
                is_qr[r] = True
                roots[r] = x
    
    # Search for the minimum s
    # s must be such that s, s+1, ..., s+4 are all in valid_residues
    # Constraints: 1 <= s <= 1004 (so s+4 <= 1008)
    # Since p=1009, s+i is in [1, 1008], so never 0 mod 1009
    
    for s in range(1, p - 4):
        if time.time() > deadline:
            break
        # Check if s, s+1, s+2, s+3, s+4 are all quadratic residues
        # Using list indexing for speed
        if is_qr[s] and is_qr[s+1] and is_qr[s+2] and is_qr[s+3] and is_qr[s+4]:
            return {
                'start': s,
                'roots': [roots[s], roots[s+1], roots[s+2], roots[s+3], roots[s+4]]
            }
            
    # Fallback (should not happen for p=1009)
    raise RuntimeError("No run found")
