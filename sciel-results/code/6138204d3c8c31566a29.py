"""Find the minimum s such that s, s+1, ..., s+4 are all non-zero quadratic residues mod 1009."""
import time

# EVOLVE-BLOCK-START
def construct():
    p = 1009
    m = 5
    deadline = time.time() + 85.0  # safety margin before 90s kill

    # Precompute: for each value v in 1..p-1, store a root r such that r^2 % p == v.
    # We want the smallest s (1 <= s <= p-5) such that s, s+1, ..., s+m-1 are all
    # non-zero quadratic residues mod p.
    #
    # Strategy:
    # 1. Build a set of quadratic residues (non-zero) mod p.
    # 2. Build an array is_qr[v] = True if v is a non-zero QR mod p.
    # 3. For each s from 1 to p-m, check if is_qr[s], is_qr[s+1], ..., is_qr[s+m-1] are all True.
    #    Return the first such s with its roots.
    #
    # This is O(p) to build residues, O(p*m) to search, which is trivial for p=1009.

    # Build the set of non-zero quadratic residues and their roots
    roots = {}
    for x in range(1, p):
        r2 = (x * x) % p
        if r2 > 0 and r2 not in roots:
            roots[r2] = x

    # Build a boolean array for quick lookup
    is_qr = [False] * p
    for r2 in roots:
        is_qr[r2] = True

    # Search for the minimum s
    for s in range(1, p - m + 1):
        # Check if s, s+1, ..., s+m-1 are all non-zero QRs
        ok = True
        for i in range(m):
            v = s + i
            if v >= p or v <= 0 or not is_qr[v]:
                ok = False
                break
        if ok:
            res_roots = [roots[s + i] for i in range(m)]
            # Verify correctness
            for i in range(m):
                assert (res_roots[i] * res_roots[i]) % p == (s + i) % p
            return {'start': s, 'roots': res_roots}

    # Fallback: should not happen for p=1009, m=5
    raise RuntimeError('No run of 5 consecutive non-zero QRs found')
# EVOLVE-BLOCK-END
