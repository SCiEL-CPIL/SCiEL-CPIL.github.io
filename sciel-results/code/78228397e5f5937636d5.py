"""Find the least run-start s for 5 consecutive non-zero quadratic residues mod 1009."""

def construct():
    P = 1009
    M = 5

    # Precompute: for each x in [1, P-1], compute x^2 mod P.
    # Since P is prime and x != 0, x^2 mod P is in [1, P-1].
    # We need the smallest root for each residue (to ensure 1 <= ri <= 1008, 
    # which is automatically satisfied, but we want deterministic minimal roots).
    
    # root_of[r] = smallest x in [1, P-1] such that x*x % P == r
    root_of = [0] * P  # index by residue, 0 means not a QR or not yet found
    
    for x in range(1, P):
        r = (x * x) % P
        if root_of[r] == 0:
            root_of[r] = x
    
    # Ascending scan: find minimum s in [1, 1004] such that s..s+4 are all non-zero QRs.
    for s in range(1, P - M + 1):
        # Check all 5 consecutive values are non-zero quadratic residues
        ok = True
        for i in range(M):
            if root_of[s + i] == 0:
                ok = False
                break
        if ok:
            return {'start': s, 'roots': [root_of[s + i] for i in range(M)]}
    
    raise RuntimeError('No valid run of 5 consecutive non-zero QRs found')
