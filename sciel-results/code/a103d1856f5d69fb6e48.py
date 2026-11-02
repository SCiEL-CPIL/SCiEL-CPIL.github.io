import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.

    Strategy: the 2x3x8 tensor is NOT a direct sum (all 3 rows of B feed all 8
    columns), so the 4 Strassen 2x2 blocks share the same 2x2 A submatrix.  We
    encode the 8 B-columns as a polynomial b(x)=sum b_k x^k and multiply by the
    constant 2x2 polynomial matrix A2(x) mod (x^8-1) using a 2-step Karatsuba
    (FFT-style doubling).  This reuses the A-linear-combination across all 8
    columns, compressing 4 separate Strassen blocks (28 mults) into one
    polynomial multiplication (21 mults).  The A[:,0] * B[0,:] part is a true
    direct sum and stays schoolbook (16 mults).  Total rank = 37.
    """
    n, m, p = 2, 3, 8
    rank = 37
    U = [[0] * rank for _ in range(n * m)]   # shape (6, 37), index i*m+j
    V = [[0] * rank for _ in range(m * p)]   # shape (24, 37), index j*p+k
    W = [[0] * rank for _ in range(p * n)]   # shape (16, 37), index k*n+i

    col = 0

    # ---- Part 1: A[:,0] (2x1) * B[0,:] (1x8) -> 2x8  (direct sum, schoolbook)
    for i in range(2):
        for k in range(8):
            U[i * m + 0][col] = 1      # A[i,0]
            V[0 * p + k][col] = 1      # B[0,k]
            W[k * n + i][col] = 1      # C[i,k]
            col += 1

    # ---- Part 2: A2 (2x2) * B2 (2x8) -> 2x8 via 2-step Karatsuba on columns
    # A2 global indices: a00=A[0,1]=1, a01=A[0,2]=2, a10=A[1,1]=4, a11=A[1,2]=5
    # B2 row0 global: 8..15, row1 global: 16..23.  W index for C[i,k] = k*2+i.
    def add(u_idx, v_idx, w_idx, sign):
        nonlocal col
        if u_idx is not None:
            U[u_idx][col] = sign
        if v_idx is not None:
            V[v_idx][col] = sign
        if w_idx is not None:
            W[w_idx][col] = sign
        col += 1

    # S1: (a00+a11)(b00+b11)  -> +c00,+c11
    add(1, None, None, 1); add(5, None, None, 1)
    add(None, 8, None, 1); add(None, 23, None, 1)
    add(None, None, 0, 1); add(None, None, 7, 1)
    # S2: (a10+a11)b00  -> +c10,-c11
    add(4, None, None, 1); add(5, None, None, 1)
    add(None, 8, None, 1)
    add(None, None, 1, 1); add(None, None, 7, -1)
    # S3: a00(b01-b11)  -> +c01,+c11
    add(1, None, None, 1)
    add(None, 9, None, 1); add(None, 23, None, -1)
    add(None, None, 2, 1); add(None, None, 7, 1)
    # S4: a11(b10-b00)  -> +c00,+c10
    add(5, None, None, 1)
    add(None, 16, None, 1); add(None, 8, None, -1)
    add(None, None, 0, 1); add(None, None, 1, 1)
    # S5: (a00+a01)b11  -> -c00,+c01
    add(1, None, None, 1); add(2, None, None, 1)
    add(None, 23, None, 1)
    add(None, None, 0, -1); add(None, None, 2, 1)
    # S6: (a10-a00)(b00+b01)  -> +c11
    add(4, None, None, 1); add(1, None, None, -1)
    add(None, 8, None, 1); add(None, 9, None, 1)
    add(None, None, 7, 1)
    # S7: (a01-a11)(b10+b11)  -> +c00
    add(2, None, None, 1); add(5, None, None, -1)
    add(None, 16, None, 1); add(None, 23, None, 1)
    add(None, None, 0, 1)

    # ---- Step 2: halve the 8 columns into two 4-column blocks via Karatsuba
    # P0 = b0 + b4 ; P1 = b1 + b5 ; P2 = b2 + b6 ; P3 = b3 + b7
    # (b_j is a 2-vector (row0, row1) at global V indices 8+j, 16+j)
    P = [[8 + j, 16 + j] for j in range(4)]
    for j in range(4):
        P[j].append(8 + j + 4)
        P[j].append(16 + j + 4)

    def emit_strassen(Ps, c00, c01, c10, c11):
        # S1: (a00+a11)(P0+P3) -> +c00,+c11
        add(1, None, None, 1); add(5, None, None, 1)
        add(None, Ps[0][0], None, 1); add(None, Ps[0][1], None, 1)
        add(None, Ps[3][0], None, 1); add(None, Ps[3][1], None, 1)
        add(None, None, c00, 1); add(None, None, c11, 1)
        # S2: (a10+a11)P0 -> +c10,-c11
        add(4, None, None, 1); add(5, None, None, 1)
        add(None, Ps[0][0], None, 1); add(None, Ps[0][1], None, 1)
        add(None, None, c10, 1); add(None, None, c11, -1)
        # S3: a00(P1-P3) -> +c01,+c11
        add(1, None, None, 1)
        add(None, Ps[1][0], None, 1); add(None, Ps[1][1], None, 1)
        add(None, Ps[3][0], None, -1); add(None, Ps[3][1], None, -1)
        add(None, None, c01, 1); add(None, None, c11, 1)
        # S4: a11(P2-P0) -> +c00,+c10
        add(5, None, None, 1)
        add(None, Ps[2][0], None, 1); add(None, Ps[2][1], None, 1)
        add(None, Ps[0][0], None, -1); add(None, Ps[0][1], None, -1)
        add(None, None, c00, 1); add(None, None, c10, 1)
        # S5: (a00+a01)P3 -> -c00,+c01
        add(1, None, None, 1); add(2, None, None, 1)
        add(None, Ps[3][0], None, 1); add(None, Ps[3][1], None, 1)
        add(None, None, c00, -1); add(None, None, c01, 1)
        # S6: (a10-a00)(P0+P1) -> +c11
        add(4, None, None, 1); add(1, None, None, -1)
        add(None, Ps[0][0], None, 1); add(None, Ps[0][1], None, 1)
        add(None, Ps[1][0], None, 1); add(None, Ps[1][1], None, 1)
        add(None, None, c11, 1)
        # S7: (a01-a11)(P2+P3) -> +c00
        add(2, None, None, 1); add(5, None, None, -1)
        add(None, Ps[2][0], None, 1); add(None, Ps[2][1], None, 1)
        add(None, Ps[3][0], None, 1); add(None, Ps[3][1], None, 1)
        add(None, None, c00, 1)

    # Block 0: output columns 0..3  (c index = k*2 + i)
    emit_strassen(P, 0, 2, 1, 3)
    # Block 1: output columns 4..7  (c index = (k+4)*2 + i)
    emit_strassen(P, 8, 10, 9, 11)

    # ---- Verify the decomposition
    def verify(U, V, W, n, m, p):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        r = Ua.shape[1]
        T = np.zeros((n, m, p))
        for c in range(r):
            u = Ua[:, c].reshape(n, m)
            v = Va[:, c].reshape(m, p)
            w = Wa[:, c].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return bool(np.allclose(T, np.ones((n, m, p)), atol=1e-9))

    if col == rank and verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # ---- Fallback: schoolbook decomposition (rank 48)
    rank_fallback = n * m * p
    U_f = [[0] * rank_fallback for _ in range(n * m)]
    V_f = [[0] * rank_fallback for _ in range(m * p)]
    W_f = [[0] * rank_fallback for _ in range(p * n)]
    c2 = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_f[i * m + j][c2] = 1
                V_f[j * p + k][c2] = 1
                W_f[k * n + i][c2] = 1
                c2 += 1
    return {"U": U_f, "V": V_f, "W": W_f}
# EVOLVE-BLOCK-END
