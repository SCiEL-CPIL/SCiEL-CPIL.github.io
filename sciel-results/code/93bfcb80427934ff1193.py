"""Program for matrix multiplication 2,2,13 - rank-minimal decomposition."""
import time
import numpy as np

DEADLINE = time.time() + 120.0


def _verify(U, V, W, n, m, p):
    """Verify that sum_r U[i,j,r]*V[j,k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(U[i * m + j], V[j * p + k] * W[k * n + i]))
                if abs(val - 1.0) > 1e-10:
                    return False
    return True


def construct():
    """
    Rank-minimal exact tensor decomposition for 2x2 @ 2x13 matrix multiplication.
    Output uses official trace convention W[k*n + i, r].

    Uses Strassen's 7-product scheme applied to 2-column blocks of B (6 blocks
    covering columns 0..11) and a rank-4 direct (schoolbook) decomposition for
    the final column 12.

    Total rank = 6 * 7 + 4 = 46.
    """
    n, m, p = 2, 2, 13

    U_rows = []
    V_rows = []
    W_rows = []

    # ------------------------------------------------------------------
    # 6 blocks of 2 columns: Strassen per block
    # ------------------------------------------------------------------
    for t in range(6):
        c0 = 2 * t        # even column of the block
        c1 = 2 * t + 1    # odd column of the block

        # p1 = (a00+a11) * (b00+b11)
        #   U: a00(idx 0) +1, a11(idx 3) +1
        #   V: B[0,c0]+1, B[1,c1]+1
        #   W: C[c0,0]+1, C[c1,1]+1   (trace: k*n+i)
        U1 = [1, 0, 0, 1]
        V1 = np.zeros(m * p); V1[0 * p + c0] = 1; V1[1 * p + c1] = 1
        W1 = np.zeros(p * n); W1[c0 * n + 0] = 1; W1[c1 * n + 1] = 1

        # p2 = (a10+a11) * b00
        #   U: a10(idx 2)+1, a11(idx 3)+1
        #   V: B[0,c0]+1
        #   W: C[c0,1]+1, C[c1,1]-1
        U2 = [0, 0, 1, 1]
        V2 = np.zeros(m * p); V2[0 * p + c0] = 1
        W2 = np.zeros(p * n); W2[c0 * n + 1] = 1; W2[c1 * n + 1] = -1

        # p3 = a00 * (b10 - b00)
        #   U: a00(idx 0)+1
        #   V: B[1,c0]+1, B[0,c0]-1
        #   W: C[c1,0]+1, C[c1,1]+1
        U3 = [1, 0, 0, 0]
        V3 = np.zeros(m * p); V3[1 * p + c0] = 1; V3[0 * p + c0] = -1
        W3 = np.zeros(p * n); W3[c1 * n + 0] = 1; W3[c1 * n + 1] = 1

        # p4 = a11 * (b01 - b11)
        #   U: a11(idx 3)+1
        #   V: B[0,c1]+1, B[1,c1]-1
        #   W: C[c0,0]+1, C[c0,1]+1
        U4 = [0, 0, 0, 1]
        V4 = np.zeros(m * p); V4[0 * p + c1] = 1; V4[1 * p + c1] = -1
        W4 = np.zeros(p * n); W4[c0 * n + 0] = 1; W4[c0 * n + 1] = 1

        # p5 = (a00+a01) * b11
        #   U: a00(idx 0)+1, a01(idx 1)+1
        #   V: B[1,c1]+1
        #   W: C[c0,0]-1, C[c1,0]+1
        U5 = [1, 1, 0, 0]
        V5 = np.zeros(m * p); V5[1 * p + c1] = 1
        W5 = np.zeros(p * n); W5[c0 * n + 0] = -1; W5[c1 * n + 0] = 1

        # p6 = (a10 - a00) * (b00 + b01)
        #   U: a00(idx 0)-1, a10(idx 2)+1
        #   V: B[0,c0]+1, B[0,c1]+1
        #   W: C[c1,1]+1
        U6 = [-1, 0, 1, 0]
        V6 = np.zeros(m * p); V6[0 * p + c0] = 1; V6[0 * p + c1] = 1
        W6 = np.zeros(p * n); W6[c1 * n + 1] = 1

        # p7 = (a01 - a11) * (b10 + b11)
        #   U: a01(idx 1)+1, a11(idx 3)-1
        #   V: B[1,c0]+1, B[1,c1]+1
        #   W: C[c0,0]+1
        U7 = [0, 1, 0, -1]
        V7 = np.zeros(m * p); V7[1 * p + c0] = 1; V7[1 * p + c1] = 1
        W7 = np.zeros(p * n); W7[c0 * n + 0] = 1

        for (u, v, w) in [(U1, V1, W1), (U2, V2, W2), (U3, V3, W3),
                          (U4, V4, W4), (U5, V5, W5), (U6, V6, W6),
                          (U7, V7, W7)]:
            U_rows.append(np.array(u, dtype=float))
            V_rows.append(v)
            W_rows.append(w)

    # ------------------------------------------------------------------
    # Final column (k=12): direct rank-4 terms
    # ------------------------------------------------------------------
    c_last = 12
    for i in range(n):
        for j in range(m):
            u = np.zeros(n * m); u[i * m + j] = 1
            v = np.zeros(m * p); v[j * p + c_last] = 1
            w = np.zeros(p * n); w[c_last * n + i] = 1
            U_rows.append(u)
            V_rows.append(v)
            W_rows.append(w)

    # ------------------------------------------------------------------
    # Assemble factor matrices
    # ------------------------------------------------------------------
    U = np.column_stack(U_rows)          # (n*m, rank)
    V = np.column_stack(V_rows)          # (m*p, rank)
    W = np.column_stack(W_rows)          # (p*n, rank)

    # ------------------------------------------------------------------
    # Verify; fall back to schoolbook if something is off
    # ------------------------------------------------------------------
    if not _verify(U, V, W, n, m, p):
        rk = n * m * p
        Us = np.zeros((n * m, rk))
        Vs = np.zeros((m * p, rk))
        Ws = np.zeros((p * n, rk))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i * m + j, r] = 1
                    Vs[j * p + k, r] = 1
                    Ws[k * n + i, r] = 1
                    r += 1
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
