import time

def construct():
    """
    Finds the minimum start s such that s, s+1, s+2, s+3, s+4 are
    non-zero quadratic residues modulo 1009.
    Returns a dictionary with 'start' and 'roots'.
    """
    t0 = time.time()
    deadline = t0 + 55.0  # safety margin before 60s kill

    p = 1009

    # Precompute quadratic residues and one root for each non-zero residue
    # is_res[i] = True if i is a non-zero QR mod p
    # root[i] = x such that x*x % p == i
    is_res = bytearray(p)  # 0/1 flags, compact
    root = [0] * p

    for x in range(1, p):
        r = (x * x) % p
        if not is_res[r]:
            is_res[r] = 1
            root[r] = x

    # Search for the minimum s
    # s must satisfy 1 <= s <= 1004 so that s+4 <= 1008 (non-zero, no wraparound)
    for s in range(1, 1005):
        if time.time() > deadline:
            # Safety: should never trigger for this small problem, but guard anyway
            raise RuntimeError("Time limit exceeded")

        # Check if s, s+1, s+2, s+3, s+4 are all residues
        if (is_res[s] and is_res[s+1] and is_res[s+2] and
                is_res[s+3] and is_res[s+4]):
            return {
                'start': s,
                'roots': [root[s], root[s+1], root[s+2], root[s+3], root[s+4]]
            }

    raise RuntimeError("No solution found")
