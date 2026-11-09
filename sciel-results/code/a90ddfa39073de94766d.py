import time

# EVOLVE-BLOCK-START
def construct():
    """
    Find the smallest s in 1..1004 such that s, s+1, s+2, s+3, s+4 are all
    quadratic residues modulo 1009 (non-zero), and return the start and roots.
    
    Uses a boolean array for O(1) residue lookup and a deadline guard.
    """
    deadline = time.monotonic() + 80.0
    p = 1009
    m = 5
    
    # Build a boolean array where is_qr[v] is True iff v is a non-zero
    # quadratic residue mod p, and also record the smallest root for each.
    is_qr = [False] * p
    smallest_root = [0] * p
    
    # x and p-x give the same residue; smallest root is min(x, p-x)
    # Iterate x from 1 to (p-1)//2 to get the smaller representative
    for x in range(1, (p - 1) // 2 + 1):
        r = (x * x) % p
        if r != 0:
            is_qr[r] = True
            smallest_root[r] = x
    
    # Scan for the first run of m consecutive QRs
    # s ranges from 1 to p - m (inclusive), i.e., 1..1004
    for s in range(1, p - m + 1):
        if time.monotonic() > deadline:
            # Safety: should never trigger for this fixed problem, but guard anyway
            raise RuntimeError('Deadline exceeded during search')
        # Check if s, s+1, ..., s+m-1 are all QRs
        ok = True
        for i in range(m):
            if not is_qr[s + i]:
                ok = False
                break
        if ok:
            roots = [smallest_root[s + i] for i in range(m)]
            return {'start': s, 'roots': roots}
    
    raise RuntimeError('No run of 5 consecutive quadratic residues found')
# EVOLVE-BLOCK-END
