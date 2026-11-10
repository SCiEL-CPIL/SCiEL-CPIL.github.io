"""Rewritten program: find the minimum s such that s, s+1, ..., s+4 are all non-zero quadratic residues mod 1009."""
import time

# EVOLVE-BLOCK-START
def construct():
    p = 1009
    m = 5
    deadline = time.time() + 85.0  # safety margin before 90s kill

    # Precompute: for each value v in 1..p-1, store a root r such that r^2 % p == v.
    # We want the smallest s (1 <= s <= 1004) such that s, s+1, ..., s+4 are all
    # non-zero quadratic residues mod p.
    #
    # Strategy:
    # 1. Build a boolean array is_res[v] = True if v is a non-zero QR mod p.
    # 2. For each v in 1..p-1, record one root.
    # 3. Scan s from 1 upward; the first s where is_res[s..s+4] are all True (with
    #    wraparound excluded, i.e., s+4 <= p-1 = 1008) is the answer.
    #
    # Since p=1009 is small, this is extremely fast.

    # is_res[v] for v in 0..p-1
    is_res = [False] * p
    root_of = [0] * p  # root_of[v] = some r with r^2 % p == v, 0 if not a residue

    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            is_res[r] = True
            if root_of[r] == 0:  # keep the smallest root (first encountered)
                root_of[r] = x

    # Scan s from 1 to p-1-m+1 = 1005 (since s+4 <= 1008 => s <= 1004)
    # Actually s ranges 1..1004 so that s+4 <= 1008.
    for s in range(1, p - m + 1):
        # Check if s, s+1, ..., s+m-1 are all non-zero QRs
        # Note: s+i must be in 1..p-1; since s >= 1 and s+m-1 <= p-1, this holds.
        ok = True
        for i in range(m):
            if not is_res[s + i]:
                ok = False
                break
        if ok:
            roots = [root_of[s + i] for i in range(m)]
            return {'start': s, 'roots': roots}

    # Should not reach here for p=1009, m=5
    raise RuntimeError('No run of 5 consecutive non-zero QRs found')
# EVOLVE-BLOCK-END
