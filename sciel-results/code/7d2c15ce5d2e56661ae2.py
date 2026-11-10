"""Find the minimum s such that s, s+1, ..., s+4 are all non-zero quadratic residues mod 1009."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    p = 1009
    m = 5
    deadline = time.time() + 85.0  # safety margin before 90s kill

    # Build a bitset (as a NumPy uint8 array) of quadratic residues mod p.
    # qr[v] = 1 if v is a non-zero quadratic residue mod p, else 0.
    # Index 0 is unused (we exclude zero residues).
    qr = np.zeros(p, dtype=np.uint8)
    roots = {}  # value -> smallest root r such that r^2 % p == value
    for x in range(1, p):
        r2 = (x * x) % p
        qr[r2] = 1
        if r2 not in roots:
            roots[r2] = x

    # We need to find the smallest s in [1, p-m] such that
    # qr[s] == qr[s+1] == qr[s+2] == qr[s+3] == qr[s+4] == 1.
    #
    # Using NumPy: create shifted views and AND them together.
    # mask[i] = 1 if qr[i] & qr[i+1] & ... & qr[i+4] all 1.
    # Then find the first index s where mask[s] == 1.
    
    # Build the AND of 5 consecutive positions
    mask = qr.copy()
    for shift in range(1, m):
        mask = mask & qr[shift:]
    
    # mask[i] corresponds to position i in the original qr array,
    # but only for indices i where i+4 < p, i.e., i < p-4 = 1005.
    # We need s >= 1, so look at mask[1:1005] (indices 1 through 1004).
    # Actually mask has length p - (m-1) = 1009 - 4 = 1005.
    # mask[0] corresponds to s=0 (excluded since s>=1).
    # mask[1] corresponds to s=1, ..., mask[1004] corresponds to s=1004.
    
    # Find the first index in mask[1:] that is 1
    # mask has shape (1005,), valid s range is 1..1004 which is mask[1:1005]
    candidates = np.where(mask[1:] == 1)[0]
    
    if len(candidates) > 0:
        s = int(candidates[0]) + 1  # add 1 because we sliced from index 1
        result_roots = [roots[s + i] for i in range(m)]
        return {'start': s, 'roots': result_roots}
    
    # Fallback: linear scan (should not be needed)
    for s in range(1, p - m + 1):
        if time.time() > deadline:
            break
        if all(qr[s + i] == 1 for i in range(m)):
            result_roots = [roots[s + i] for i in range(m)]
            return {'start': s, 'roots': result_roots}
    
    raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found')
# EVOLVE-BLOCK-END
