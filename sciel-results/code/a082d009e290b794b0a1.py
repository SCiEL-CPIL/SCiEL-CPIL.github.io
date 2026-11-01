"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen's decomposition of rank 7 for 2x2 matrix multiplication.
    n, m, p = DIMENSIONS
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # m1 = (a11 + a22)(b11 + b22)
    U[0][0] = 1  # a11
    U[3][0] = 1  # a22
    V[0][0] = 1  # b11
    V[3][0] = 1  # b22
    W[0][0] = 1  # c11 (k=0, i=0)
    W[3][0] = 1  # c22 (k=1, i=1)
    
    # m2 = (a21 + a22)b11
    U[2][1] = 1  # a21
    U[3][1] = 1  # a22
    V[0][1] = 1  # b11
    W[2][1] = 1  # c21 (k=1, i=0)
    
    # m3 = a11(b12 - b22)
    U[0][2] = 1  # a11
    V[1][2] = 1  # b12
    V[3][2] = -1 # b22
    W[1][2] = 1  # c12 (k=0, i=1)
    W[3][2] = -1 # c22
    
    # m4 = a22(b21 - b11)
    U[3][3] = 1  # a22
    V[2][3] = 1  # b21
    V[0][3] = -1 # b11
    W[0][3] = 1  # c11
    W[2][3] = 1  # c21
    
    # m5 = (a11 + a12)b22
    U[0][4] = 1  # a11
    U[1][4] = 1  # a12
    V[3][4] = 1  # b22
    W[0][4] = -1 # c11
    W[1][4] = 1  # c12
    
    # m6 = (a21 - a11)(b11 + b12)
    U[2][5] = 1  # a21
    U[0][5] = -1 # a11
    V[0][5] = 1  # b11
    V[1][5] = 1  # b12
    W[3][5] = 1  # c22
    
    # m7 = (a12 - a22)(b21 + b22)
    U[1][6] = 1  # a12
    U[3][6] = -1 # a22
    V[2][6] = 1  # b21
    V[3][6] = 1  # b22
    W[0][6] = -1 # c11
    W[3][6] = 1  # c22

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
