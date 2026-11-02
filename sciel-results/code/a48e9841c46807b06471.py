import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    start_time = time.time()
    deadline = start_time + 140.0

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(s - (1.0 if j == jp else 0.0)) > tol:
                            return False
        return True

    # ---- Known rank-8 decomposition of the 2x2x3 multiplication tensor ----
    # U: (i,j) indexed i*2+j,  V: (j',k) indexed j'*3+k,  W: (k,i) indexed k*2+i
    U_223 = np.array([
        [ 1,  0,  0,  0,  1,  0,  0,  1],   # i=0,j=0
        [ 0,  1,  0,  0,  0,  0,  0,  0],   # i=0,j=1
        [ 0,  0,  1,  0,  0,  1,  1,  0],   # i=1,j=0
        [ 0,  0,  0,  1,  0,  0,  0,  1],   # i=1,j=1
    ], dtype=np.int32)
    V_223 = np.array([
        [ 1,  0,  0,  0,  0,  0,  0,  0],   # j'=0,k=0
        [ 0,  0,  0,  0,  1,  0,  1,  0],   # j'=0,k=1
        [ 0,  0,  0,  0,  0,  1,  0,  0],   # j'=0,k=2
        [ 0,  1,  0,  0,  0,  0,  0,  0],   # j'=1,k=0
        [ 0,  0,  1,  0,  0,  0,  0,  0],   # j'=1,k=1
        [ 0,  0,  0,  1,  0,  0,  0,  0],   # j'=1,k=2
    ], dtype=np.int32)
    W_223 = np.array([
        [ 1,  0,  0,  0,  0,  0,  0,  0],   # k=0,i=0
        [ 0,  0,  0,  0,  1,  0,  1,  0],   # k=0,i=1
        [ 0,  0,  0,  0,  0,  1,  0,  1],   # k=1,i=0
        [ 0,  0,  0,  0,  0,  0,  0,  1],   # k=1,i=1
        [ 0,  1,  0,  0,  0,  0,  0,  0],   # k=2,i=0
        [ 0,  0, 1,  0,  0,  0,  0,  0],   # k=2,i=1
    ], dtype=np.int32)

    # Verify 2x2x3 sub-decomposition
    def verify_223():
        for i in range(2):
            for j in range(2):
                for jp in range(2):
                    for k in range(3):
                        s = 0.0
                        for c in range(8):
                            s += U_223[i*2+j, c] * V_223[jp*3+k, c] * W_223[k*2+i, c]
                        if abs(s - (1.0 if j == jp else 0.0)) > 1e-9:
                            return False
        return True

    # ---- Strassen 2x2x2 block ----
    def add_strassen_222(U, V, W, col, i0, j0, k0):
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)

        # M1
        U[a, col] += 1; U[d, col] += 1
        V[e, col] += 1; V[h, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        # M2
        U[c, col] += 1; U[d, col] += 1
        V[e, col] += 1
        W[c10, col] += 1; W[c11, col] -= 1
        col += 1
        # M3
        U[a, col] += 1
        V[f, col] += 1; V[h, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        # M4
        U[d, col] += 1
        V[g, col] += 1; V[e, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        # M5
        U[a, col] += 1; U[b, col] += 1
        V[h, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1
        # M6
        U[c, col] += 1; U[a, col] -= 1
        V[e, col] += 1; V[f, col] += 1
        W[c11, col] += 1
        col += 1
        # M7
        U[b, col] += 1; U[d, col] -= 1
        V[g, col] += 1; V[h, col] += 1
        W[c00, col] += 1
        col += 1
        return col

    # ---- Schoolbook 2x2 * 2x1 (rank 4) ----
    def add_schoolbook_221(U, V, W, col, i0, j0, k0):
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)

        U[a, col] += 1; V[e, col] += 1; W[c00, col] += 1; col += 1
        U[b, col] += 1; V[g, col] += 1; W[c00, col] += 1; col += 1
        U[c, col] += 1; V[e, col] += 1; W[c10, col] += 1; col += 1
        U[d, col] += 1; V[g, col] += 1; W[c10, col] += 1; col += 1
        return col

    # ---- Strategy 1: 2x2x3 + 2x2x2 + 2x2x2 per j-block, total rank 44 ----
    if verify_223():
        rank = 44
        U = np.zeros((n*m, rank), dtype=np.int32)
        V = np.zeros((m*p, rank), dtype=np.int32)
        W = np.zeros((p*n, rank), dtype=np.int32)
        col = 0
        for j_start in [0, 2]:
            # 2x2x3 block for k=0,1,2
            for c in range(8):
                for i in range(2):
                    for j in range(2):
                        U[i*m + (j_start + j), col + c] += U_223[i*2 + j, c]
                for jp in range(2):
                    for k in range(3):
                        V[(j_start + jp)*p + k, col + c] += V_223[jp*3 + k, c]
                for k in range(3):
                    for i in range(2):
                        W[k*n + i, col + c] += W_223[k*2 + i, c]
            col += 8
            # Two Strassen 2x2x2 blocks for k=3,4 and k=5,6
            col = add_strassen_222(U, V, W, col, 0, j_start, 3)
            col = add_strassen_222(U, V, W, col, 0, j_start, 5)
        if col == rank and verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ---- Strategy 2: all Strassen + one schoolbook per j-block, total rank 50 ----
    rank = 50
    U = np.zeros((n*m, rank), dtype=np.int32)
    V = np.zeros((m*p, rank), dtype=np.int32)
    W = np.zeros((p*n, rank), dtype=np.int32)
    col = 0
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            col = add_strassen_222(U, V, W, col, 0, j_start, k_start)
        col = add_schoolbook_221(U, V, W, col, 0, j_start, 6)
    if col == rank and verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ---- Ultimate fallback: full schoolbook, rank n*m*p = 56 ----
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int32)
    V = np.zeros((m*p, rank), dtype=np.int32)
    W = np.zeros((p*n, rank), dtype=np.int32)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1
                V[j*p + k, idx] = 1
                W[k*n + i, idx] = 1
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
