import time


def construct():
    """
    Find the minimum s in [1, 1004] such that s, s+1, s+2, s+3, s+4
    are all non-zero quadratic residues modulo 1009.

    Returns a dict with 'start' (the minimum s) and 'roots' (list of 5 integers,
    one per residue, such that root**2 % 1009 equals the residue).
    """
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin

    p = 1009
    m = 5  # run length

    # Precompute a root for each non-zero quadratic residue modulo p.
    # root_map[r] = x where x*x % p == r, for r in [1, p-1] that are QRs.
    root_map = [0] * p
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and root_map[r] == 0:
            root_map[r] = x

    # Scan s from 1 to p - m (= 1004), looking for the first s where
    # s, s+1, ..., s+m-1 are all non-zero QRs (i.e., root_map[val] != 0).
    for s in range(1, p - m + 1):
        if time.time() > deadline:
            break

        # Check all m consecutive values
        all_residue = True
        for offset in range(m):
            val = s + offset
            if root_map[val] == 0:
                all_residue = False
                break

        if all_residue:
            roots = [root_map[s + offset] for offset in range(m)]
            return {
                'start': s,
                'roots': roots
            }

    # Fallback: should not occur for p=1009, but handle gracefully.
    raise RuntimeError("No valid run of 5 consecutive non-zero QRs found modulo 1009")
