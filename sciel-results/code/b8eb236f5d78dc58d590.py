import time

def construct():
    # Deadline guard with safety margin
    start_time = time.time()
    deadline = 50.0

    p = 1009
    m = 5  # number of consecutive residues needed

    # Structural change: use a boolean array instead of bitmask
    # is_residue[r] = True if r is a non-zero quadratic residue mod p
    # root[r] = one square root of r mod p
    is_residue = [False] * p
    root = [0] * p

    for x in range(1, p):
        r = (x * x) % p
        if not is_residue[r]:
            is_residue[r] = True
            root[r] = x

    # Scan for the smallest s in [1, p-m] such that s, s+1, ..., s+m-1
    # are all non-zero quadratic residues
    s = None
    for candidate in range(1, p - m + 1):
        # Deadline check
        if time.time() - start_time > deadline:
            break
        if all(is_residue[candidate + i] for i in range(m)):
            s = candidate
            break

    if s is None:
        raise ValueError("No run of 5 consecutive non-zero QRs found")

    result_roots = [root[s + i] for i in range(m)]
    return {'start': s, 'roots': result_roots}
