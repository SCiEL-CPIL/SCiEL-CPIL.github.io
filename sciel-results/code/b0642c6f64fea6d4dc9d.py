import time
import numpy as np

def construct():
    _t0 = time.time()
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        # 7 Strassen products for a 2x2 * 2x2 block
        # X_rows = [A11, A12, A21, A22] (row indices in U)
        # Y_rows = [B11, B12, B21, B22] (row indices in V)
        # Z_rows = [C11, C12, C21, C22] (row indices in W)
        # M1 = (A11+A22)(B11+B22)
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        # M2 = (A21+A22)B11
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        # M3 = A11(B21-B22)
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1
        # M4 = A22(B12-B11)
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1
        # M5 = (A11+A12)B22
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1
        # M6 = (A21-A11)(B11+B12)
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1
        # M7 = (A12-A22)(B21+B22)
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    # Four 2x2 block multiplications
    # A is 2x4: A = [A11 A12; A21 A22] where A11,A12,A21,A22 are 2x2
    # B is 4x4: B = [B11 B12; B21 B22] where B11,B12,B21,B22 are 2x2
    # C = A*B is 2x4: C = [C11 C12; C21 C22]

    # Block (0,0): A11 * B11 -> C11
    add_block(0, [0, 1, 4, 5], [0, 1, 4, 5], [0, 2, 1, 3])
    # Block (1,0): A21 * B11 -> C21
    add_block(7, [2, 3, 6, 7], [8, 9, 12, 13], [0, 2, 1, 3])
    # Block (0,1): A12 * B21 -> C12
    add_block(14, [0, 1, 4, 5], [2, 3, 6, 7], [4, 6, 5, 7])
    # Block (1,1): A22 * B21 -> C22
    add_block(21, [2, 3, 6, 7], [10, 11, 14, 15], [4, 6, 5, 7])

    def verify_solution(U_mat, V_mat, W_mat, r):
        n, m, p = 2, 4, 4
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

    if not verify_solution(U, V, W, 28):
        # Fallback: construct schoolbook rank-32 solution
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

    _elapsed = time.time() - _t0
    if _elapsed > 110:
        pass
    return {"U": U, "V": V, "W": W}
