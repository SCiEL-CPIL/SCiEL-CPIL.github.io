"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 11
    U = [[0] * rank for _ in range(n * m)]  # 4 x 11
    V = [[0] * rank for _ in range(m * p)]  # 6 x 11
    W = [[0] * rank for _ in range(p * n)]  # 6 x 11

    half = 0.5

    # Terms 1-7: Landsberg rank-7 decomposition for 2x2x2 (k=0,1)
    # U index: i*m+j = i*2+j
    # V index: j*p+k = j*3+k
    # W index: k*n+i = k*2+i

    # Term 1: A[0,0], B[0,0], C[0,0]
    U[0][0] = half
    V[0][0] = 1
    W[0][0] = 1

    # Term 2: A[1,1], B[1,1], C[1,1]
    U[3][1] = half
    V[4][1] = 1
    W[3][1] = 1

    # Term 3: A[0,1], B[1,0], C[0,0]
    U[1][2] = half
    V[3][2] = 1
    W[0][2] = 1

    # Term 4: A[1,0], B[0,1], C[1,1]
    U[2][3] = half
    V[1][3] = 1
    W[3][3] = 1

    # Term 5: A[0,0]+A[0,1], B[1,1], C[0,1]
    U[0][4] = half
    U[1][4] = half
    V[4][4] = 1
    W[2][4] = 1

    # Term 6: A[1,0]+A[1,1], B[0,0], C[1,0]
    U[2][5] = half
    U[3][5] = half
    V[0][5] = 1
    W[1][5] = 1

    # Term 7: A[0,0]+A[1,1], B[0,1]+B[1,0], C[0,1]+C[1,0]
    U[0][6] = half
    U[3][6] = half
    V[1][6] = 1
    V[3][6] = 1
    W[2][6] = 1
    W[1][6] = 1

    # Terms 8-11: Schoolbook for k=2 (2x2x1 tensor)
    # Term 8: A[0,0], B[0,2], C[0,2]
    U[0][7] = half
    V[2][7] = 1
    W[4][7] = 1

    # Term 9: A[0,1], B[1,2], C[0,2]
    U[1][8] = half
    V[5][8] = 1
    W[4][8] = 1

    # Term 10: A[1,0], B[0,2], C[1,2]
    U[2][9] = half
    V[2][9] = 1
    W[5][9] = 1

    # Term 11: A[1,1], B[1,2], C[1,2]
    U[3][10] = half
    V[5][10] = 1
    W[5][10] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
