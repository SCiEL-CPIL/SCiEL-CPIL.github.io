import time

P = 1009
M = 5

def construct():
    # Precompute: is_qr[v] = True if v is a nonzero QR mod P
    # root[v] = a root x (1 <= x <= P-1) such that x^2 % P == v
    is_qr = [False] * P
    root = [0] * P

    for x in range(1, P):
        r = (x * x) % P
        if r != 0:
            is_qr[r] = True
            if root[r] == 0:
                root[r] = x

    # Search for minimal s in [1, P-M] such that s..s+4 are all nonzero QRs
    for s in range(1, P - M + 1):
        if (is_qr[s] and is_qr[s + 1] and is_qr[s + 2]
                and is_qr[s + 3] and is_qr[s + 4]):
            roots = [root[s], root[s + 1], root[s + 2], root[s + 3], root[s + 4]]
            # Validate each root
            for i in range(M):
                assert (roots[i] * roots[i]) % P == s + i, \
                    f"Root validation failed: {roots[i]}^2 mod {P} != {s+i}"
                assert 1 <= roots[i] <= P - 1
            return {'start': s, 'roots': roots}

    # Fallback: should never be reached for P=1009, M=5
    raise RuntimeError("No run of 5 consecutive nonzero QRs found mod 1009")
