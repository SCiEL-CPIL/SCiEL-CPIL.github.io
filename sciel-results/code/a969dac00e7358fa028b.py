"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    p = 1009
    m = 5
    deadline = time.time() + 85.0  # safety margin before 90s limit

    # Use Legendre symbol via vectorized modular exponentiation to identify QRs.
    # (a/p) = a^((p-1)/2) mod p  ->  1 if QR, p-1 if non-QR, 0 if a==0
    x = np.arange(1, p, dtype=np.int64)
    # Compute x^((p-1)/2) mod p using numpy's pow with modular reduction
    # numpy doesn't have built-in pow with mod for arrays in older versions,
    # so we do it in a loop-free way: use repeated squaring manually
    # Actually, for p=1009, (p-1)/2 = 504. We can just compute x^504 % p.
    # NumPy doesn't have vectorized pow with mod, so we'll use a small loop
    # or just compute x*x % p and check if it's a perfect square... 
    # Better: just compute the squares directly since p is small.
    
    # Direct approach: compute all squares, mark residues
    residues = (x * x) % p
    qr_mask = np.zeros(p, dtype=bool)
    qr_mask[residues] = True
    qr_mask[0] = False  # exclude zero
    
    # Build roots: for each residue, store one valid root (smallest x)
    roots = np.zeros(p, dtype=np.int32)
    for i in range(len(x)):
        r = residues[i]
        if roots[r] == 0:
            roots[r] = x[i]
    
    # Find minimal s in [1, p-m] such that s..s+m-1 are all QR
    # Use sliding window via cumsum of non-QR indicators
    # non_qr[i] = 1 if i is NOT a QR (or i==0), else 0
    non_qr = (~qr_mask).astype(np.int32)
    # We need sum of non_qr[s..s+m-1] == 0
    # Use cumsum: prefix[i] = sum(non_qr[0..i-1])
    prefix = np.zeros(p + 1, dtype=np.int32)
    np.cumsum(non_qr, out=prefix[1:])
    
    # For s in [1, p-m], check prefix[s+m] - prefix[s] == 0
    candidates = np.arange(1, p - m + 1)
    window_sums = prefix[candidates + m] - prefix[candidates]
    valid = np.where(window_sums == 0)[0]
    
    if len(valid) == 0:
        # Fallback: direct search
        for s in range(1, p - m + 1):
            if all(qr_mask[s + i] for i in range(m)):
                candidates_valid = [s]
                break
        else:
            raise RuntimeError('No run of 5 consecutive non-zero QRs found')
        best_s = candidates_valid[0]
    else:
        best_s = int(candidates[valid[0]])
    
    result_roots = [int(roots[best_s + i]) for i in range(m)]
    
    return {'start': best_s, 'roots': result_roots}
# EVOLVE-BLOCK-END
