import numpy as np

def construct():
    p = 1009
    # Compute quadratic residues via vectorized squaring (x=1..1008)
    xs = np.arange(1, p, dtype=np.int64)
    rs = (xs * xs) % p

    # Build membership array for residues 1..1008 (index 0 unused for value 0)
    # is_qr[v] is True iff v is a quadratic residue mod p and v != 0
    is_qr = np.zeros(p, dtype=bool)
    is_qr[rs] = True
    is_qr[0] = False  # exclude zero

    # Find the smallest s in [1, 1004] such that s, s+1, s+2, s+3, s+4 are all QRs
    # We check via sliding window of size 5 on the boolean array
    # For s from 1 to 1004, we need is_qr[s:s+5].all()
    # Vectorized: create a 2D view? Simpler to just loop since p is small,
    # but let's use a cumulative approach or direct vectorized check.

    # Direct vectorized check:
    # For each s in range(1, 1005), check if all 5 consecutive are True
    s_candidates = np.arange(1, p - 3)  # 1 to 1004
    # We need is_qr[s], is_qr[s+1], ..., is_qr[s+4]
    # Stack these:
    checks = np.stack([is_qr[s_candidates + i] for i in range(5)], axis=1)
    valid = np.all(checks, axis=1)

    # Find first valid s
    valid_indices = np.where(valid)[0]
    if len(valid_indices) == 0:
        raise RuntimeError("No run of 5 consecutive quadratic residues found")
    
    s = int(s_candidates[valid_indices[0]])

    # Now find roots for s, s+1, s+2, s+3, s+4
    # We need any x in [1, 1008] such that x^2 mod p == s+i
    # Build a mapping from residue to one root
    root_map = {}
    for x, r in zip(xs, rs):
        r_int = int(r)
        if r_int not in root_map:
            root_map[r_int] = int(x)

    res_roots = [root_map[s + i] for i in range(5)]

    return {'start': s, 'roots': res_roots}
