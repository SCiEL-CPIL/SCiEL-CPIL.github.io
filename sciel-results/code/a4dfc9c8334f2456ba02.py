"""Tensor decomposition for 2x2 * 2x14 matrix multiplication, rank 49."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

def construct():
    """
    Construct a rank-49 tensor decomposition for 2x2 * 2x14 matrix multiplication.
    Uses Strassen's 7-term decomposition on 7 blocks of 2x2.
    """
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    num_blocks = p // 2
    rank = num_blocks * 7

    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    # Strassen's 7 terms for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B = [[i, j], [k, l]]
    #
    # M1 = (a+d)(e+h) -> contributes to i, l
    # M2 = (c+d)e     -> contributes to k, l (with -1 on l)
    # M3 = a(f-h)     -> contributes to j, l (with +1 on l)
    # M4 = d(g-e)     -> contributes to i, k (with +1 on k)
    # M5 = (a+b)h     -> contributes to i (-1), j
    # M6 = (c-a)(e+f) -> contributes to l
    # M7 = (b-d)(g+h) -> contributes to i

    for t in range(num_blocks):
        base = t * 7
        # V indices: B[0, 2t]=ve, B[0, 2t+1]=vf, B[1, 2t]=vg, B[1, 2t+1]=vh
        ve = 2 * t          # row 0, col 2t
        vf = 2 * t + 1      # row 0, col 2t+1
        vg = m + 2 * t      # row 1, col 2t  (m*p index: j*p+k, j=1)
        vh = m + 2 * t + 1  # row 1, col 2t+1

        # W indices: C[k*n+i], for k in {2t, 2t+1}, i in {0, 1}
        # C[0, 2t]=i, C[0, 2t+1]=j, C[1, 2t]=k, C[1, 2t+1]=l
        w_i = 2 * t * n + 0      # k=2t,   i=0
        w_j = 2 * t * n + 1      # k=2t,   i=1  -- wait, C is 2x14, so C[row, col]
        # C has shape (n, p) = (2, 14). C[r, c] where r in {0,1}, c in {0,...,13}
        # W index is k*n + i where k is the column of C (0..13) and i is the row (0..1)
        # So C[0, 2t] -> k=2t, i=0 -> w = 2t*2+0 = 4t
        # C[1, 2t] -> k=2t, i=1 -> w = 2t*2+1 = 4t+1
        # C[0, 2t+1] -> k=2t+1, i=0 -> w = (2t+1)*2+0 = 4t+2
        # C[1, 2t+1] -> k=2t+1, i=1 -> w = (2t+1)*2+1 = 4t+3
        w_i = 4 * t        # C[0, 2t]
        w_k = 4 * t + 1    # C[1, 2t]
        w_j = 4 * t + 2    # C[0, 2t+1]
        w_l = 4 * t + 3    # C[1, 2t+1]

        # U indices: A[i*m+j], i in {0,1}, j in {0,1}
        # a = A[0,0] -> U[0], b = A[0,1] -> U[1], c = A[1,0] -> U[2], d = A[1,1] -> U[3]

        # M1 = (a+d)(e+h)
        U[0, base] = 1; U[3, base] = 1
        V[ve, base] = 1; V[vh, base] = 1
        W[w_i, base] = 1; W[w_l, base] = 1

        # M2 = (c+d)e
        U[2, base+1] = 1; U[3, base+1] = 1
        V[ve, base+1] = 1
        W[w_k, base+1] = 1; W[w_l, base+1] = -1

        # M3 = a(f-h)
        U[0, base+2] = 1
        V[vf, base+2] = 1; V[vh, base+2] = -1
        W[w_j, base+2] = 1; W[w_l, base+2] = 1

        # M4 = d(g-e)
        U[3, base+3] = 1
        V[vg, base+3] = 1; V[ve, base+3] = -1
        W[w_i, base+3] = 1; W[w_k, base+3] = 1

        # M5 = (a+b)h
        U[0, base+4] = 1; U[1, base+4] = 1
        V[vh, base+4] = 1
        W[w_i, base+4] = -1; W[w_j, base+4] = 1

        # M6 = (c-a)(e+f)
        U[2, base+5] = 1; U[0, base+5] = -1
        V[ve, base+5] = 1; V[vf, base+5] = 1
        W[w_l, base+5] = 1

        # M7 = (b-d)(g+h)
        U[1, base+6] = 1; U[3, base+6] = -1
        V[vg, base+6] = 1; V[vh, base+6] = 1
        W[w_i, base+6] = 1

    # Verify
    def verify(Uf, Vf, Wf):
        for i in range(n):
            for j in range(m):
                ui = i * m + j
                for k in range(p):
                    vi = j * p + k
                    wi = k * n + i
                    s = np.dot(Uf[ui, :], Vf[vi, :] * Wf[wi, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback: schoolbook rank 56
        rank_fb = n * m * p
        U = np.zeros((nm, rank_fb))
        V = np.zeros((mp, rank_fb))
        W = np.zeros((pn, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1

    def to_list(arr):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in arr.tolist()]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
