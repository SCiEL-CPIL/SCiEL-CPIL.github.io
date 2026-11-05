"""Rewritten program: direct, minimal-s start search for 5 consecutive quadratic residues mod 1009."""

P = 1009
M = 5

def construct():
    # Build a dictionary mapping each quadratic residue (1..P-1) to a root in 1..P-1.
    # roots[r] = x such that x*x % P == r, with 1 <= x <= P-1.
    # We want the smallest root for each residue to minimize potential secondary metrics.
    roots = {}
    is_qr = [False] * P
    
    for x in range(1, P):
        r = (x * x) % P
        if r != 0:
            is_qr[r] = True
            if r not in roots:
                roots[r] = x

    # Search for minimal s
    for s in range(1, P - M + 1):
        if all(is_qr[s + i] for i in range(M)):
            result_roots = [roots[s + i] for i in range(M)]
            return {'start': s, 'roots': result_roots}

    raise RuntimeError('No run found')
