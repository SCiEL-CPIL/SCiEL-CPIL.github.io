"""Find the minimal start s for five consecutive non-zero quadratic residues mod 1009.
Uses a boolean array for O(1) residue lookup and ascending scan for minimal s.
Includes a wall-clock deadline guard for safety."""
# EVOLVE-BLOCK-START
import time

_P = 1009
_RUN = 5


def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # safety margin under the 90s kill

    # Build a boolean array is_qr of length p (index 0..p-1)
    # and a root_map storing one witness root for each non-zero QR.
    is_qr = [False] * _P
    root_map = {}

    for x in range(1, _P):
        r = (x * x) % _P
        is_qr[r] = True
        if r not in root_map:
            root_map[r] = x

    # Ascending scan: first 5 consecutive residues => minimal s.
    # s ranges 1..1004, so s+4 <= 1008 < p; no wraparound or zero.
    for s in range(1, _P - _RUN + 1):
        if time.time() > deadline:
            break
        if is_qr[s] and is_qr[s + 1] and is_qr[s + 2] and is_qr[s + 3] and is_qr[s + 4]:
            roots = [root_map[s + i] for i in range(_RUN)]
            return {'start': s, 'roots': roots}

    raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found')
# EVOLVE-BLOCK-END
