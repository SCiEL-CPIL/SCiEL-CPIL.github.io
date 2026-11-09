"""Find the minimal start s for 5 consecutive quadratic residues modulo 1009."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    p = 1009
    m = 5

    # Precompute: for each non-zero residue r in 1..p-1, store one root x
    # such that x*x % p == r. Since p is prime, each non-zero QR has exactly
    # two roots (x and p-x), so we just need one.
    root_of = [0] * p  # root_of[r] = a root x for residue r, 0 if not a QR
    for x in range(1, p):
        if time.time() > deadline:
            break
        r = (x * x) % p
        if r != 0 and root_of[r] == 0:
            root_of[r] = x

    # Search for the minimum s in 1..p-m such that s, s+1, ..., s+m-1
    # are all quadratic residues (non-zero).
    best_result = None
    for s in range(1, p - m + 1):
        if time.time() > deadline:
            break
        ok = True
        for i in range(m):
            if root_of[s + i] == 0:
                ok = False
                break
        if ok:
            roots = [root_of[s + i] for i in range(m)]
            best_result = {'start': s, 'roots': roots}
            # Since we iterate s from 1 upwards, the first valid s is the minimum.
            # We can return immediately.
            return best_result

    if best_result is not None:
        return best_result

    # Fallback (should not occur for p=1009, m=5)
    raise RuntimeError('No run of 5 consecutive QRs found')
# EVOLVE-BLOCK-END
