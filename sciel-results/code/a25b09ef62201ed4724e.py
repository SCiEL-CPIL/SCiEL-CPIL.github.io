import time
import sys

def construct():
    p = 1009
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin from 90s limit
    
    # Precompute quadratic residues and their smallest roots
    # is_qr[i] is True if i is a non-zero QR mod p
    # root_map[i] is the smallest x in [1, p-1] such that x^2 % p == i
    is_qr = [False] * p
    root_map = {}
    
    for x in range(1, p):
        r = (x * x) % p
        is_qr[r] = True
        if r not in root_map:
            root_map[r] = x
            
    # Search for minimum s
    # s must be in [1, p-5] to ensure s+4 <= p-1
    # Also s+i must be >= 1
    for s in range(1, p - 4):
        if time.time() > deadline:
            # Return best found so far? 
            # Since we search in order, if we haven't found one, we have none.
            # But s=1 is expected. If we timeout, something is wrong.
            # We can break and raise or return a fallback.
            pass
            
        # Check if s, s+1, s+2, s+3, s+4 are all QRs
        # Since we start from 1, the first valid s is the minimum.
        valid = True
        for i in range(5):
            val = s + i
            if not is_qr[val]:
                valid = False
                break
        
        if valid:
            roots = [root_map[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}
            
    # If no run found (should not happen for p=1009, m=5)
    raise RuntimeError("No run of 5 consecutive QRs found")
