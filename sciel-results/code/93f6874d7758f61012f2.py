P = 1009
M = 5

def construct():
    # Precompute: for each residue r in 1..P-1, find a root x with x^2 ≡ r (mod P)
    # We only need one root per residue.
    root_of = [0] * P
    for x in range(1, P):
        r = (x * x) % P
        if r != 0 and root_of[r] == 0:
            root_of[r] = x

    # For each s in 1..P-M, check if s, s+1, ..., s+M-1 are all non-zero QR
    # We can do this by checking root_of[s+i] != 0 for all i
    for s in range(1, P - M + 1):
        ok = True
        for i in range(M):
            if root_of[s + i] == 0:
                ok = False
                break
        if ok:
            roots = [root_of[s + i] for i in range(M)]
            # Validate
            for i in range(M):
                assert (roots[i] * roots[i]) % P == s + i
            return {'start': s, 'roots': roots}

    raise RuntimeError("No run found")
