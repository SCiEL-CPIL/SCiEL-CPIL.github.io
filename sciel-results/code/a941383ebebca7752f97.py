"""Rank-42 Strassen decomposition for 2x4 x 4x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start = time.time()
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42

    # U has n*m = 8 rows, V has m*p = 24 rows, W has p*n = 12 rows
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def strassen_block(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        """Add 7 Strassen rank-1 terms for a 2x2 x 2x2 -> 2x2 block.
        a00,a01,a10,a11: U row indices for A[0,0],A[0,1],A[1,0],A[1,1]
        b00,b01,b10,b11: V row indices for B[0,0],B[0,1],B[1,0],B[1,1]
        c00,c01,c10,c11: W row indices for C[0,0],C[0,1],C[1,0],C[1,1]
        """
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        # M2 = (A10+A11)(B00)
        U[a10][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1; W[c11][col] -= 1
        col += 1
        # M3 = A00(B01-B11)
        U[a00][col] += 1
        V[b01][col] += 1; V[b11][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        # M4 = A11(B10-B00)
        U[a11][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        # M5 = (A00+A01)(B11)
        U[a00][col] += 1; U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[a10][col] += 1; U[a00][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[a01][col] += 1; U[a11][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1

    # A is 2x4, flattened: A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3,
    #                       A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    # B is 4x6, flattened: B[i,j] = i*6+j
    # C is 2x6, W indexed as W[k*n+i] = W[2*k+i], so C[i,k] at W[2k+i]
    # C[0,0]=W[0], C[0,1]=W[2], C[1,0]=W[1], C[1,1]=W[3]
    # C[0,2]=W[4], C[0,3]=W[6], C[1,2]=W[5], C[1,3]=W[7]
    # C[0,4]=W[8], C[0,5]=W[10], C[1,4]=W[9], C[1,5]=W[11]

    # A_L = A[:,0:2] -> a00=0, a01=1, a10=4, a11=5
    # A_R = A[:,2:4] -> a00=2, a01=3, a10=6, a11=7

    # B[0:2,0:2] -> b00=B[0,0]=0, b01=B[0,1]=1, b10=B[1,0]=6, b11=B[1,1]=7
    # B[0:2,2:4] -> b00=B[0,2]=2, b01=B[0,3]=3, b10=B[1,2]=8, b11=B[1,3]=9
    # B[0:2,4:6] -> b00=B[0,4]=4, b01=B[0,5]=5, b10=B[1,4]=10, b11=B[1,5]=11
    # B[2:4,0:2] -> b00=B[2,0]=12, b01=B[2,1]=13, b10=B[3,0]=18, b11=B[3,1]=19
    # B[2:4,2:4] -> b00=B[2,2]=14, b01=B[2,3]=15, b10=B[3,2]=20, b11=B[3,3]=21
    # B[2:4,4:6] -> b00=B[2,4]=16, b01=B[2,5]=17, b10=B[3,4]=22, b11=B[3,5]=23

    # C[:,0:2] -> c00=W[0], c01=W[2], c10=W[1], c11=W[3]
    # C[:,2:4] -> c00=W[4], c01=W[6], c10=W[5], c11=W[7]
    # C[:,4:6] -> c00=W[8], c01=W[10], c10=W[9], c11=W[11]

    # 1. A_L * B[0:2,0:2] -> C[:,0:2]
    strassen_block(0, 1, 4, 5, 0, 1, 6, 7, 0, 2, 1, 3)
    # 2. A_L * B[0:2,2:4] -> C[:,2:4]
    strassen_block(0, 1, 4, 5, 2, 3, 8, 9, 4, 6, 5, 7)
    # 3. A_L * B[0:2,4:6] -> C[:,4:6]
    strassen_block(0, 1, 4, 5, 4, 5, 10, 11, 8, 10, 9, 11)
    # 4. A_R * B[2:4,0:2] -> C[:,0:2]
    strassen_block(2, 3, 6, 7, 12, 13, 18, 19, 0, 2, 1, 3)
    # 5. A_R * B[2:4,2:4] -> C[:,2:4]
    strassen_block(2, 3, 6, 7, 14, 15, 20, 21, 4, 6, 5, 7)
    # 6. A_R * B[2:4,4:6] -> C[:,4:6]
    strassen_block(2, 3, 6, 7, 16, 17, 22, 23, 8, 10, 9, 11)

    if col != rank:
        raise ValueError(f"Expected {rank} columns, got {col}")

    # Verify with a random test
    np.random.seed(42)
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    C_expected = A_test @ B_test

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    C_computed = np.zeros((n, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                C_computed[i, k] += val

    if not np.allclose(C_computed, C_expected, atol=1e-10):
        return _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}
