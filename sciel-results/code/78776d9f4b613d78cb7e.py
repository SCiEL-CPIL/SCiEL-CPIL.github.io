"""Find the minimum start s such that s, s+1, ..., s+4 are all quadratic residues mod 1009."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    t0 = time.time()
    deadline = t0 + 88  # safety margin before 90s wall-clock limit

    # Build a boolean array is_res[v] for v in 0..p-1 (True iff v is a non-zero QR mod p)
    # and store one square root for each non-zero residue in roots[v].
    is_res = [False] * p
    roots = [0] * p
    for x in range(1, p):
        r = (x * x) % p
        is_res[r] = True
        if roots[r] == 0:
            roots[r] = x
    # is_res[0] stays False -> zero residues excluded

    # Ascending scan for the smallest s in 1..1004 such that s..s+4 are all non-zero QRs.
    # Use a sliding window sum over the is_res array to test each candidate in O(1).
    # Window w(s) = sum_{i=0..4} is_res[s+i]. w(s) == 5 iff all five are residues.
    # Compute w(1) initially, then slide.
    # s ranges 1..1004 so that s+4 <= 1008 (no wraparound, no zero).

    # Initial window at s=1: indices 1,2,3,4,5
    w = sum(is_res[1:6])
    if w == 5:
        s = 1
        return {'start': s, 'roots': [roots[s + i] for i in range(5)]}

    for s in range(2, p - 3):  # s from 2 to 1004 inclusive
        # Slide: remove is_res[s-1], add is_res[s+4]
        w -= is_res[s - 1]
        w += is_res[s + 4]
        if w == 5:
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
        if time.time() > deadline:
            raise RuntimeError('Time limit exceeded')

    raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found')
# EVOLVE-BLOCK-END
