"""Strassen-based block decomposition for 4x4 * 4x9 matrix multiplication.
Splits B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1).
C = A*B1 + A*B2 + A*B3
Rank: 7 (Strassen) + 7 (Strassen) + 16 (schoolbook) = 30
"""
import time

DIMENSIONS = [4, 4, 9]

def _strassen_4x4x4():
    """
    Return U (16 x 7), V (16 x 7), W (16 x 7) for 4x4 * 4x4.
    U[i*4+j, c] = coefficient for A[i,j]
    V[j*4+k, c] = coefficient for B[j,k]
    W[k*4+i, c] = coefficient for C[i,k]  (output index = k*n+i, n=4)
    """
    U = [[0.0]*7 for _ in range(16)]
    V = [[0.0]*7 for _ in range(16)]
    W = [[0.0]*7 for _ in range(16)]

    def block_entries(bi, bj):
        entries = []
        for lr in range(2):
            for lc in range(2):
                r = 2*bi + lr
                c = 2*bj + lc
                entries.append((r, c))
        return entries

    # M1 = (A11+A22)(B11+B22)
    for (r, c) in block_entries(0, 0):
        U[r*4+c][0] += 1.0
    for (r, c) in block_entries(1, 1):
        U[r*4+c][0] += 1.0
    for (r, c) in block_entries(0, 0):
        V[r*4+c][0] += 1.0
    for (r, c) in block_entries(0, 1):
        V[r*4+c][0] += 1.0

    # M2 = (A21+A22)B11
    for (r, c) in block_entries(1, 0):
        U[r*4+c][1] += 1.0
    for (r, c) in block_entries(1, 1):
        U[r*4+c][1] += 1.0
    for (r, c) in block_entries(0, 0):
        V[r*4+c][1] += 1.0

    # M3 = A11(B12-B22)
    for (r, c) in block_entries(0, 0):
        U[r*4+c][2] += 1.0
    for (r, c) in block_entries(0, 1):
        V[r*4+c][2] += 1.0
    for (r, c) in block_entries(1, 1):
        V[r*4+c][2] -= 1.0

    # M4 = A22(B21-B11)
    for (r, c) in block_entries(1, 1):
        U[r*4+c][3] += 1.0
    for (r, c) in block_entries(1, 0):
        V[r*4+c][3] += 1.0
    for (r, c) in block_entries(0, 0):
        V[r*4+c][3] -= 1.0

    # M5 = (A11+A12)B22
    for (r, c) in block_entries(0, 0):
        U[r*4+c][4] += 1.0
    for (r, c) in block_entries(0, 1):
        U[r*4+c][4] += 1.0
    for (r, c) in block_entries(1, 1):
        V[r*4+c][4] += 1.0

    # M6 = (A21-A11)(B11+B12)
    for (r, c) in block_entries(1, 0):
        U[r*4+c][5] += 1.0
    for (r, c) in block_entries(0, 0):
        U[r*4+c][5] -= 1.0
    for (r, c) in block_entries(0, 0):
        V[r*4+c][5] += 1.0
    for (r, c) in block_entries(0, 1):
        V[r*4+c][5] += 1.0

    # M7 = (A12-A22)(B21+B22)
    for (r, c) in block_entries(0, 1):
        U[r*4+c][6] += 1.0
    for (r, c) in block_entries(1, 1):
        U[r*4+c][6] -= 1.0
    for (r, c) in block_entries(1, 0):
        V[r*4+c][6] += 1.0
    for (r, c) in block_entries(1, 1):
        V[r*4+c][6] += 1.0

    # W: output C[i,k], W index = k*4+i
    # C11 = M1+M4-M5+M7  -> C[0:2, 0:2], i in 0-1, k in 0-1
    for (r, c) in block_entries(0, 0):
        i, k = r, c
        W[k*4+i][0] += 1.0   # M1
        W[k*4+i][3] += 1.0   # M4
        W[k*4+i][4] -= 1.0   # -M5
        W[k*4+i][6] += 1.0   # M7

    # C12 = M3+M5 -> C[0:2, 2:4], i in 0-1, k in 2-3
    for (r, c) in block_entries(0, 1):
        i, k = r, c
        W[k*4+i][2] += 1.0   # M3
        W[k*4+i][4] += 1.0   # M5

    # C21 = M2+M4 -> C[2:4, 0:2], i in 2-3, k in 0-1
    for (r, c) in block_entries(1, 0):
        i, k = r, c
        W[k*4+i][1] += 1.0   # M2
        W[k*4+i][3] += 1.0   # M4

    # C22 = M1-M2+M3+M6 -> C[2:4, 2:4], i in 2-3, k in 2-3
    for (r, c) in block_entries(1, 1):
        i, k = r, c
        W[k*4+i][0] += 1.0   # M1
        W[k*4+i][1] -= 1.0   # -M2
        W[k*4+i][2] += 1.0   # M3
        W[k*4+i][5] += 1.0   # M6

    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    t0 = time.time()

    # Split B (4x9) into B1 (4x4, cols 0-3), B2 (4x4, cols 4-7), B3 (4x1, col 8)
    # C = A*B1 + A*B2 + A*B3
    # Rank: 7 + 7 + 16 = 30

    U1, V1, W1 = _strassen_4x4x4()
    U2, V2, W2 = _strassen_4x4x4()

    # Schoolbook for 4x4 * 4x1
    U3 = [[0.0]*16 for _ in range(16)]
    V3 = [[0.0]*16 for _ in range(4)]
    W3 = [[0.0]*16 for _ in range(4)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(1):
                U3[i*4+j][col] = 1.0
                V3[j*1+k][col] = 1.0
                W3[k*4+i][col] = 1.0
                col += 1

    rank = 30
    U = [[0.0]*rank for _ in range(n*m)]   # 16 x 30
    V = [[0.0]*rank for _ in range(m*p)]   # 36 x 30
    W = [[0.0]*rank for _ in range(p*n)]   # 36 x 30

    # Block 1: B1 is cols 0-3 of B
    for c in range(7):
        for row in range(16):
            U[row][c] = U1[row][c]
        for r in range(16):
            j = r // 4
            k = r % 4
            V[j*p + k][c] = V1[r][c]
        for r in range(16):
            k = r // 4
            i = r % 4
            W[k*n + i][c] = W1[r][c]

    # Block 2: B2 is cols 4-7 of B
    offset2 = 7
    for c in range(7):
        for row in range(16):
            U[row][offset2 + c] = U2[row][c]
        for r in range(16):
            j = r // 4
            k_local = r % 4
            V[j*p + 4 + k_local][offset2 + c] = V2[r][c]
        for r in range(16):
            k_local = r // 4
            i = r % 4
            W[(4 + k_local)*n + i][offset2 + c] = W2[r][c]

    # Block 3: B3 is col 8 of B
    offset3 = 14
    for c in range(16):
        for row in range(16):
            U[row][offset3 + c] = U3[row][c]
        for r in range(4):
            j = r
            V[j*p + 8][offset3 + c] = V3[r][c]
        for r in range(4):
            i = r
            W[8*n + i][offset3 + c] = W3[r][c]

    # Convert to integers
    def to_int(matrix):
        result = []
        for row in matrix:
            new_row = []
            for val in row:
                if abs(val - round(val)) < 1e-9:
                    new_row.append(int(round(val)))
                else:
                    new_row.append(val)
            result.append(new_row)
        return result

    U = to_int(U)
    V = to_int(V)
    W = to_int(W)

    return {"U": U, "V": V, "W": W}
