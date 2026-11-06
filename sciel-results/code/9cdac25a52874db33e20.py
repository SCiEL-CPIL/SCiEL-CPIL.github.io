"""Find the minimal start s for a run of 5 quadratic residues modulo 1009."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    p = 1009

    # Precompute quadratic residues and their roots
    root_map = {}
    is_qr = [False] * p

    for x in range(1, p):
        r = (x * x) % p
        is_qr[r] = True
        if r not in root_map:
            root_map[r] = x

    # Find the smallest s in [1, 1004] such that
    # s, s+1, s+2, s+3, s+4 are all non-zero quadratic residues.
    # Since s <= 1004, s+4 <= 1008, so all indices are valid.

    for s in range(1, 1005):
        if all(is_qr[s + i] for i in range(5)):
            roots = [root_map[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}

    # Fallback: should not occur for p=1009
    raise RuntimeError("No valid run of 5 consecutive quadratic residues found")
# EVOLVE-BLOCK-END
