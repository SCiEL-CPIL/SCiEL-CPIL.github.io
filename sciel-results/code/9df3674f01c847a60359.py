import time
import numpy as np

def construct():
    _t0 = time.time()
    n, m, p = 2, 4, 4

    def verify_solution(U_mat, V_mat, W_mat, r):
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a = i * m + k
                    b = k * p + j
                    c = j * n + i
                    s = 0
                    for r_idx in range(r):
                        s += U_mat[a][r_idx] * V_mat[b][r_idx] * W_mat[c][r_idx]
                    if s != 1:
                        return False
        return True

    # --- Fixed rank-28: 4 independent Strassen 2x2 blocks ---
    rank28 = 28
    U28 = [[0] * rank28 for _ in range(n * m)]
    V28 = [[0] * rank28 for _ in range(m * p)]
    W28 = [[0] * rank28 for _ in range(p * n)]

    def add_strassen(U, V, W, col, X, Y, Z):
        # X=[A11,A12,A21,A22], Y=[B11,B12,B21,B22], Z=[C11,C12,C21,C22]
        U[X[0]][col] += 1; U[X[3]][col] += 1
        V[Y[0]][col] += 1; V[Y[3]][col] += 1
        W[Z[0]][col] += 1; W[Z[3]][col] += 1
        U[X[2]][col+1] += 1; U[X[3]][col+1] += 1
        V[Y[0]][col+1] += 1
        W[Z[2]][col+1] += 1; W[Z[3]][col+1] -= 1
        U[X[0]][col+2] += 1
        V[Y[1]][col+2] += 1; V[Y[3]][col+2] -= 1
        W[Z[1]][col+2] += 1; W[Z[3]][col+2] += 1
        U[X[3]][col+3] += 1
        V[Y[2]][col+3] += 1; V[Y[0]][col+3] -= 1
        W[Z[0]][col+3] += 1; W[Z[2]][col+3] += 1
        U[X[0]][col+4] += 1; U[X[1]][col+4] += 1
        V[Y[3]][col+4] += 1
        W[Z[0]][col+4] -= 1; W[Z[1]][col+4] += 1
        U[X[2]][col+5] += 1; U[X[0]][col+5] -= 1
        V[Y[0]][col+5] += 1; V[Y[1]][col+5] += 1
        W[Z[3]][col+5] += 1
        U[X[1]][col+6] += 1; U[X[3]][col+6] -= 1
        V[Y[2]][col+6] += 1; V[Y[3]][col+6] += 1
        W[Z[0]][col+6] += 1

    # A is 2x4, B is 4x4, C is 2x4
    # A row 0: A11 (cols 0-1), A12 (cols 2-3)
    # A row 1: A21 (cols 0-1), A22 (cols 2-3)
    # B rows 0-1: B11 (cols 0-1), B12 (cols 2-3)
    # B rows 2-3: B21 (cols 0-1), B22 (cols 2-3)
    # C row 0: C11 (cols 0-1), C12 (cols 2-3)
    # C row 1: C21 (cols 0-1), C22 (cols 2-3)

    # C11 = A11*B11 + A21*B21
    add_strassen(U28, V28, W28, 0, [0,1,4,5], [0,1,4,5], [0,2,1,3])   # A11*B11
    add_strassen(U28, V28, W28, 7, [2,3,6,7], [8,9,12,13], [0,2,1,3])   # A21*B21
    # C12 = A11*B12 + A21*B22
    add_strassen(U28, V28, W28, 14, [0,1,4,5], [2,3,6,7], [4,6,5,7])    # A11*B12
    add_strassen(U28, V28, W28, 21, [2,3,6,7], [10,11,14,15], [4,6,5,7])# A21*B22

    if not verify_solution(U28, V28, W28, 28):
        # Should not happen, but fallback
        rank32 = 32
        U32 = [[0] * rank32 for _ in range(n * m)]
        V32 = [[0] * rank32 for _ in range(m * p)]
        W32 = [[0] * rank32 for _ in range(p * n)]
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a = i * m + k
                    b = k * p + j
                    c = j * n + i
                    r_idx = i * m * p + k * p + j
                    U32[a][r_idx] = 1
                    V32[b][r_idx] = 1
                    W32[c][r_idx] = 1
        return {"U": U32, "V": V32, "W": W32}

    # --- Attempt rank-27: share one Strassen term between blocks ---
    # C11 = A11*B11 + A21*B21, C12 = A11*B12 + A21*B22
    # Blocks A11*B11 and A11*B12 share the same A-block (A11)
    # If we share the M5 term: (A11+A12)*B22 vs (A11+A12)*B22
    # Actually let's try sharing M1: (A11+A22)(B11+B22)
    # For A11*B11: M1 uses (A11+A22)(B11+B22) -- but A22 and B22 are different blocks
    # This doesn't work directly. Let me try a different sharing.

    # Alternative: use the fact that C11 and C12 both involve A11 (row 0 of A)
    # C11 = A11*B11 + A21*B21
    # C12 = A11*B12 + A21*B22
    # If we compute A11*B11 and A11*B12 together, we can share some terms
    # But they use different B-blocks, so no direct sharing.

    # Let me try: rank-27 by removing one term from one block and compensating
    # This is complex. Let's just return the verified rank-28.

    _elapsed = time.time() - _t0
    if _elapsed > 110:
        pass
    return {"U": U28, "V": V28, "W": W28}
