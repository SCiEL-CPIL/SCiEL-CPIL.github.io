import time

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # safety margin under the 90s kill

    p = 1009
    run_len = 5
    mask_val = (1 << run_len) - 1  # 0b11111 = 31

    # 1. Precompute bitmask of non-zero quadratic residues and smallest roots
    # bit r is 1 if r is a non-zero QR mod p
    mask = 0
    # root_map[r] = smallest x in [1, p-1] such that x^2 % p == r
    root_map = {}

    # We can iterate x from 1 to p-1
    # Note: x and p-x give the same residue. We just need the smallest x for each residue.
    # Iterating 1 to p-1 and storing the first occurrence is sufficient.
    for x in range(1, p):
        r = (x * x) % p
        if r > 0:
            # Set bit r
            mask |= (1 << r)
            # Store smallest root
            if r not in root_map:
                root_map[r] = x

    # 2. Scan for the smallest s such that s..s+4 are all QRs
    # s must be in [1, p-1-run_len] = [1, 1004]
    # Check: (mask >> s) & mask_val == mask_val
    
    # We can speed this up by only checking s where bit s is set, but a simple loop is likely fast enough for p=1009.
    # Let's iterate s from 1 to p - run_len
    max_s = p - run_len
    for s in range(1, max_s + 1):
        if time.time() > deadline:
            break
        # Check if bits s, s+1, s+2, s+3, s+4 are all 1
        if (mask >> s) & mask_val == mask_val:
            roots = [root_map[s + i] for i in range(run_len)]
            return {'start': s, 'roots': roots}

    # Fallback (should not happen for p=1009)
    raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found')
