"""Matrix multiplication 4x4x4 rank-minimal decomposition."""
DIMENSIONS = [4, 4, 4]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-49 decomposition for 4x4 matrix multiplication
    via the tensor (Kronecker) product of the rank-7 Strassen scheme.
    This is the known minimal rank for this dimension in the Kronecker family.
    """
    start = time.time()
    limit = start + 140

    n, m, p = DIMENSIONS

    # Strassen 2x2x2 factor matrices (4,7)
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # M0: (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    # M1: (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    # M2: A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    # M3: A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    # M4: (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    # M5: (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    # M6: (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    r = 49
    U = np.zeros((n * m, r), dtype=int)
    V = np.zeros((m * p, r), dtype=int)
    W = np.zeros((p * n, r), dtype=int)

    # Pre-compute nonzero indices for each row of U2, V2, W2 for sparse expansion
    U2_nz = [(row, np.nonzero(U2[row])[0]) for row in range(4)]
    V2_nz = [(row, np.nonzero(V2[row])[0]) for row in range(4)]
    W2_nz = [(row, np.nonzero(W2[row])[0]) for row in range(4)]

    # Fill U via sparse Kronecker expansion
    for ii in range(n):
        a = ii >> 1
        b = ii & 1
        for jj in range(m):
            c = jj >> 1
            d = jj & 1
            row = ii * m + jj
            idxA = U2_nz[a * 2 + c][1]
            idxB = U2_nz[b * 2 + d][1]
            for cc in idxA:
                v1 = U2[a * 2 + c, cc]
                base = cc * 7
                for dd in idxB:
                    U[row, base + dd] = v1 * U2[b * 2 + d, dd]

    if time.time() > limit:
        raise TimeoutError()

    # Fill V via sparse Kronecker expansion
    for jj in range(m):
        c = jj >> 1
        d = jj & 1
        for kk in range(p):
            e = kk >> 1
            f = kk & 1
            row = jj * p + kk
            idxA = V2_nz[c * 2 + e][1]
            idxB = V2_nz[d * 2 + f][1]
            for cc in idxA:
                v1 = V2[c * 2 + e, cc]
                base = cc * 7
                for dd in idxB:
                    V[row, base + dd] = v1 * V2[d * 2 + f, dd]

    if time.time() > limit:
        raise TimeoutError()

    # Fill W via sparse Kronecker expansion
    for kk in range(p):
        e = kk >> 1
        f = kk & 1
        for ii in range(n):
            a = ii >> 1
            b = ii & 1
            row = kk * n + ii
            idxA = W2_nz[e * 2 + a][1]
            idxB = W2_nz[f * 2 + b][1]
            for cc in idxA:
                v1 = W2[e * 2 + a, cc]
                base = cc * 7
                for dd in idxB:
                    W[row, base + dd] = v1 * W2[f * 2 + b, dd]

    if time.time() > limit:
        raise TimeoutError()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
