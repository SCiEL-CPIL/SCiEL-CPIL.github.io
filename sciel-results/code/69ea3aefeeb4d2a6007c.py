import time

def construct():
    """
    Find the minimal start s (1<=s<=1004) such that s, s+1, ..., s+4 are all
    non-zero quadratic residues modulo p=1009. Return the start and a root
    for each residue. The score is s (lower is better).
    """
    p = 1009

    # Precompute the Legendre symbol / quadratic residue status for 0..p-1.
    # is_res[i] is True iff i is a non-zero quadratic residue mod p.
    # Also store one square root for each residue value.
    is_res = [False] * p
    root_of = [None] * p

    for r in range(1, p):
        v = (r * r) % p
        if v != 0:
            if not is_res[v]:
                is_res[v] = True
                root_of[v] = r

    # The task asks for s in [1, 1004]. Since s+i must be in [1, 1008] (no wraparound
    # past p-1=1008 and no zero residue), the effective range for s is [1, 1004].
    # We scan in order to find the minimal s.

    for s in range(1, 1005):
        # Check if s, s+1, s+2, s+3, s+4 are all non-zero quadratic residues
        if (is_res[s] and is_res[s + 1] and is_res[s + 2] and
                is_res[s + 3] and is_res[s + 4]):
            roots = [root_of[s], root_of[s + 1], root_of[s + 2],
                     root_of[s + 3], root_of[s + 4]]
            return {'start': s, 'roots': roots}

    # This fallback should never be reached for p=1009 with m=5, but included for safety.
    # If no run of 5 consecutive non-zero quadratic residues is found, return a dummy
    # that will be invalid, but the loop above should always find one.
    return {'start': 1004, 'roots': [1, 1, 1, 1, 1]}
