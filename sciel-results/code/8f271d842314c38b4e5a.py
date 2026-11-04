"""
4x4x4 matrix multiplication: rank-49 tensor decomposition via block-Strassen.
Reconstructed with an index-level construction distinct from prior attempts.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _strassen_factors():
    """
    Rank-7 Strassen for 2x2.  Indexing convention:
      A is 2x2, flat index a = i*2 + j  (row i, col j)
      B is 2x2, flat index b = j*2 + k  (row j, col k)
      C is 2x2, flat index c = k*2 + i  (trace convention: col k, row i)
    Term r contributes  U[a,r]*V[b,r]*W[c,r]  to T[a,b,c].
    """
    # Each term: list of (flat_index, coefficient) for U, V, W
    terms = [
        # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        ([ (0,1),(3,1) ], [ (0,1),(3,1) ], [ (0,1),(3,1) ]),
        # M2 = (A10+A11)B00       -> C10 += 1, C11 -= 1
        ([ (2,1),(3,1) ], [ (0,1)  ], [ (1,1),(3,-1) ]),
        # M3 = A00(B01-B11)       -> C01 += 1, C11 += 1
        ([ (0,1)  ], [ (1,1),(3,-1) ], [ (1,1),(3,1) ]),
        # M4 = A11(B10-B00)       -> C00 += 1, C10 += 1
        ([ (3,1)  ], [ (2,1),(0,-1) ], [ (0,1),(1,1) ]),
        # M5 = (A00+A01)B11       -> C00 -= 1, C01 += 1
        ([ (0,1),(1,1) ], [ (3,1)  ], [ (0,-1),(1,1) ]),
        # M6 = (A10-A00)(B00+B01) -> C11 += 1
        ([ (2,1),(0,-1) ], [ (0,1),(1,1) ], [ (3,1) ]),
        # M7 = (A01-A11)(B10+B11) -> C00 += 1
        ([ (1,1),(3,-1) ], [ (2,1),(3,1) ], [ (0,1) ]),
    ]
    R = len(terms)
    U = np.zeros((4, R))
    V = np.zeros((4, R))
    W = np.zeros((4, R))
    for r, (us, vs, ws) in enumerate(terms):
        for idx, coeff in us:
            U[idx, r] = coeff
        for idx, coeff in vs:
            V[idx, r] = coeff
        for idx, coeff in ws:
            W[idx, r] = coeff
    return U, V, W


def _build_rank49_via_blocks():
    """
    4x4 = (2x2 blocks) of (2x2).
    Write i = 2*i1+i2, j = 2*j1+j2, k = 2*k1+k2.
    The multiplication tensor decomposes as a Kronecker product of two
    2x2x2 tensors.  The rank-7 Strassen factors for each level give:
      U_4[  i  ,  r  ] = U2[ i1*2+j1 , r1 ] * U2[ i2*2+j2 , r2 ]
      V_4[  j  ,  r  ] = V2[ j1*2+k1 , r1 ] * V2[ j2*2+k2 , r2 ]
      W_4[  k  ,  r  ] = W2[ k1*2+i1 , r1 ] * W2[ k2*2+i2 , r2 ]
    where r = r1*7 + r2, total rank = 49.
    """
    U2, V2, W2 = _strassen_factors()
    R2 = 7
    R4 = R2 * R2  # 49

    # Precompute reshaped factor columns for fast Kronecker assembly
    # U2 reshaped: (i1, j1, r) -> U2_3d
    U2r = U2.reshape(2, 2, R2)
    V2r = V2.reshape(2, 2, R2)
    W2r = W2.reshape(2, 2, R2)

    U4 = np.zeros((16, R4))
    V4 = np.zeros((16, R4))
    W4 = np.zeros((16, R4))

    for r1 in range(R2):
        for r2 in range(R2):
            r = r1 * R2 + r2
            # Column r of U4: Kronecker of U2[:,r1] and U2[:,r2]
            # U2[:,r1] has shape (4,) = (2,2) indexed [i1, j1]
            # U2[:,r2] has shape (4,) = (2,2) indexed [i2, j2]
            # Result: (4,4) indexed [i1*2+i2, j1*2+j2], flatten
            col_u = np.kron(U2[:, r1].reshape(2, 2),
                            U2[:, r2].reshape(2, 2))
            U4[:, r] = col_u.ravel()

            col_v = np.kron(V2[:, r1].reshape(2, 2),
                            V2[:, r2].reshape(2, 2))
            V4[:, r] = col_v.ravel()

            col_w = np.kron(W2[:, r1].reshape(2, 2),
                            W2[:, r2].reshape(2, 2))
            W4[:, r] = col_w.ravel()

    return U4, V4, W4


def _check_decomposition(U, V, W):
    """
    Verify: sum_r U[a,r] V[b,r] W[c,r] == T[a,b,c]
    where T[a,b,c] = 1 iff a=2i+j, b=2j+k, c=2k+i for some i,j,k in {0,1}.
    For 4x4: T[4i+j, 4j+k, 4k+i]... wait, dimensions are 4,4,4.
    n=m=p=4, so A is 4x4, B is 4x4, C is 4x4.
    Flat: a = i*4+j, b = j*4+k, c = k*4+i.
    T[a,b,c] = 1 when a%4 == b//4 and b%4 == c//4 and c%4 == a//4.
    Actually: T[i*4+j, j*4+k, k*4+i] = 1.
    """
    T = np.zeros((16, 16, 16))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                T[i*4+j, j*4+k, k*4+i] = 1.0
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.max(np.abs(T_decomp - T))
    return err


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 115  # 5s safety margin under 120s

    # Primary: rank-49 Kronecker-Strassen
    try:
        U, V, W = _build_rank49_via_blocks()
        err = _check_decomposition(U, V, W)
        if err < 1e-9:
            return {
                "U": U.astype(object).tolist(),
                "V": V.astype(object).tolist(),
                "W": W.astype(object).tolist(),
            }
    except Exception:
        pass

    # Fallback: schoolbook rank 64
    if time.time() > deadline:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
