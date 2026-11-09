import time

def construct():
    # Deadline guard with safety margin
    start_time = time.monotonic()
    deadline = 50.0

    p = 1009
    m = 5  # number of consecutive residues needed

    # Precompute roots: root[r] is a square root of r mod p if r is a non-zero QR, else 0
    # This is O(p) and extremely fast for p=1009
    root = [0] * p

    for x in range(1, p):
        r = (x * x) % p
        # Only store the first root found for each residue
        # Since we iterate x from 1, this is deterministic
        if root[r] == 0:
            root[r] = x

    # Scan for the smallest s in [1, p-m] such that s, s+1, ..., s+m-1
    # are all non-zero quadratic residues
    s = None
    max_s = p - m + 1
    
    for candidate in range(1, max_s):
        # Deadline check
        if time.monotonic() - start_time > deadline:
            break
        
        # Check if all m consecutive numbers are residues
        # root[val] != 0 implies val is a non-zero QR
        if all(root[candidate + i] != 0 for i in range(m)):
            s = candidate
            break

    if s is None:
        raise ValueError("No run of 5 consecutive non-zero QRs found")

    result_roots = [root[s + i] for i in range(m)]
    return {'start': s, 'roots': result_roots}
