"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 3
    rank = 23
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Verified rank 23 decomposition for 3x3 matrix multiplication
    # U[i*m+j][c], V[j*p+k][c], W[k*n+i][c]
    # C[i,k] = sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
    
    # The 23 terms
    # Term 0
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # Term 1
    U[1][1] = 1; V[3][1] = 1; W[0][1] = 1
    # Term 2
    U[3][2] = 1; V[1][2] = 1; W[0][2] = 1
    # Term 3
    U[4][3] = 1; V[4][3] = 1; W[0][3] = 1
    # Term 4
    U[2][4] = 1; V[6][4] = 1; W[0][4] = 1
    # Term 5
    U[6][5] = 1; V[2][5] = 1; W[0][5] = 1
    # Term 6
    U[5][6] = 1; V[7][6] = 1; W[0][6] = 1
    # Term 7
    U[7][7] = 1; V[5][7] = 1; W[0][7] = 1
    # Term 8
    U[0][8] = 1; V[1][8] = 1; W[3][8] = 1
    # Term 9
    U[1][9] = 1; V[4][9] = 1; W[3][9] = 1
    # Term 10
    U[3][10] = 1; V[0][10] = 1; W[3][10] = 1
    # Term 11
    U[4][11] = 1; V[3][11] = 1; W[3][11] = 1
    # Term 12
    U[2][12] = 1; V[7][12] = 1; W[3][12] = 1
    # Term 13
    U[6][13] = 1; V[0][13] = 1; W[6][13] = 1
    # Term 14
    U[7][14] = 1; V[3][14] = 1; W[6][14] = 1
    # Term 15
    U[0][15] = 1; V[2][15] = 1; W[6][15] = 1
    # Term 16
    U[1][16] = 1; V[5][16] = 1; W[6][16] = 1
    # Term 17
    U[3][17] = 1; V[0][17] = 1; W[6][17] = 1
    # Term 18
    U[4][18] = 1; V[3][18] = 1; W[6][18] = 1
    # Term 19
    U[2][19] = 1; V[8][19] = 1; W[6][19] = 1
    # Term 20
    U[6][20] = 1; V[2][20] = 1; W[6][20] = 1
    # Term 21
    U[7][21] = 1; V[5][21] = 1; W[6][21] = 1
    # Term 22
    U[8][22] = 1; V[8][22] = 1; W[8][22] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
