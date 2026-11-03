import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140  # seconds, safety margin before 150s kill

    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=4, p=8

    # Strassen decomposition for 2x2 matrix multiplication (rank 7)
    U_local = [[0]*7 for _ in range(4)]
    V_local = [[0]*7 for _ in range(4)]
    W_local = [[0]*7 for _ in range(4)]

    # r=0 (M1): (a00+a11)(b00+b11)
    U_local[0][0] = 1; U_local[3][0] = 1
    V_local[0][0] = 1; V_local[3][0] = 1
    W_local[0][0] = 1  # C00
    W_local[3][0] = 1  # C11

    # r=1 (M2): (a10+a11)b00
    U_local[2][1] = 1; U_local[3][1] = 1
    V_local[0][1] = 1
    W_local[2][1] = 1   # C10
    W_local[3][1] = -1  # C11

    # r=2 (M3): a00(b01-b11)
    U_local[0][2] = 1
    V_local[1][2] = 1; V_local[3][2] = -1
    W_local[1][2] = 1   # C01
    W_local[3][2] = 1   # C11

    # r=3 (M4): a11(b10-b00)
    U_local[3][3] = 1
    V_local[2][3] = 1; V_local[0][3] = -1
    W_local[0][3] = 1   # C00
    W_local[2][3] = 1   # C10

    # r=4 (M5): (a00+a01)b11
    U_local[0][4] = 1; U_local[1][4] = 1
    V_local[3][4] = 1
    W_local[0][4] = -1  # C00
    W_local[1][4] = 1   # C01

    # r=5 (M6): (a10-a00)(b00+b01)
    U_local[2][5] = 1; U_local[0][5] = -1
    V_local[0][5] = 1; V_local[1][5] = 1
    W_local[3][5] = 1   # C11

    # r=6 (M7): (a01-a11)(b10+b11)
    U_local[1][6] = 1; U_local[3][6] = -1
    V_local[2][6] = 1; V_local[3][6] = 1
    W_local[0][6] = 1   # C00

    total_rank = 56

    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 56

    # Decompose 2x4 x 4x8 into 8 blocks of 2x2 x 2x2
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7

            a_map = [
                0 * 4 + (2 * s + 0),
                0 * 4 + (2 * s + 1),
                1 * 4 + (2 * s + 0),
                1 * 4 + (2 * s + 1),
            ]

            b_map = [
                (2 * s + 0) * 8 + (2 * j + 0),
                (2 * s + 0) * 8 + (2 * j + 1),
                (2 * s + 1) * 8 + (2 * j + 0),
                (2 * s + 1) * 8 + (2 * j + 1),
            ]

            c_map = [
                (2 * j + 0) * 2 + 0,
                (2 * j + 1) * 2 + 0,
                (2 * j + 0) * 2 + 1,
                (2 * j + 1) * 2 + 1,
            ]

            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la][l]
                    if val != 0:
                        U[a_map[la]][r] += val
                for lb in range(4):
                    val = V_local[lb][l]
                    if val != 0:
                        V[b_map[lb]][r] += val
                for lc in range(4):
                    val = W_local[lc][l]
                    if val != 0:
                        W[c_map[lc]][r] += val

    # Check for linear dependencies among the 56 rank-1 terms
    # Construct the tensor matrix M of size (n*m*p) x 56
    M = np.zeros((n * m * p, total_rank))
    for r in range(total_rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = (i * m + j) * p + k
                    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    M[idx, r] = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]

    # Compute the rank of M
    rank_M = np.linalg.matrix_rank(M)
    
    # If rank_M < total_rank, we can reduce the number of terms
    if rank_M < total_rank:
        # Find a basis of columns
        # Use QR decomposition with column pivoting
        Q, R, P = np.linalg.qr(M, mode='complete', pivoting=True)
        # The first rank_M columns of M[:, P[:rank_M]] form a basis
        basis_cols = P[:rank_M]
        
        # Construct new U, V, W with only the basis columns
        U_new = [[U[i][r] for r in basis_cols] for i in range(n*m)]
        V_new = [[V[i][r] for r in basis_cols] for i in range(m*p)]
        W_new = [[W[i][r] for r in basis_cols] for i in range(p*n)]
        
        U = U_new
        V = V_new
        W = W_new
        total_rank = rank_M

    if time.time() - start_time > deadline:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
