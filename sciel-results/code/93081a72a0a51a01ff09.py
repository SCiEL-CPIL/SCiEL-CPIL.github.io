"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for A (2x3):
    # A00=0, A01=1, A02=2
    # A10=3, A11=4, A12=5
    
    # Indices for B (3x3):
    # B00=0, B01=1, B02=2
    # B10=3, B11=4, B12=5
    # B20=6, B21=7, B22=8
    
    # Indices for C (2x3), W[k*n+i]:
    # C00=0, C01=1
    # C10=2, C11=3
    # C20=4, C21=5
    
    # Part 1: Strassen for A_L (cols 0,1) * B_top_12 (rows 0,1, cols 0,1)
    # A_L = [[A00, A01], [A10, A11]]
    # B_top_12 = [[B00, B01], [B10, B11]]
    
    # p1 = A00 * (B10 + B11)
    U[0][0] = 1
    V[3][0] = 1
    V[4][0] = 1
    
    # p2 = (A00 + A01) * B10
    U[0][1] = 1
    U[1][1] = 1
    V[3][1] = 1
    
    # p3 = (A10 + A11) * B00
    U[3][2] = 1
    U[4][2] = 1
    V[0][2] = 1
    
    # p4 = A11 * (B01 - B00)
    U[4][3] = 1
    V[1][3] = 1
    V[0][3] = -1
    
    # p5 = (A00 + A11) * (B00 + B11)
    U[0][4] = 1
    U[4][4] = 1
    V[0][4] = 1
    V[4][4] = 1
    
    # p6 = (A01 - A11) * (B10 + B11)
    U[1][5] = 1
    U[4][5] = -1
    V[3][5] = 1
    V[4][5] = 1
    
    # p7 = (A00 - A10) * (B00 + B01)
    U[0][6] = 1
    U[3][6] = -1
    V[0][6] = 1
    V[1][6] = 1
    
    # W contributions for C[:, 0:2]
    # C00 = p5 + p4 - p2 + p6
    W[0][0] = 1
    W[0][1] = -1
    W[0][3] = 1
    W[0][4] = 1
    W[0][5] = 1
    
    # C01 = p1 + p2
    W[1][0] = 1
    W[1][1] = 1
    
    # C10 = p3 + p4
    W[2][2] = 1
    W[2][3] = 1
    
    # C11 = p5 + p1 - p3 + p7
    W[3][0] = 1
    W[3][2] = -1
    W[3][4] = 1
    W[3][6] = 1
    
    # Part 2: Schoolbook for A_L (cols 0,1) * B_top_3 (rows 0,1, col 2)
    # B_top_3 = [[B02], [B12]]
    
    # s1 = A00 * B02
    U[0][7] = 1
    V[2][7] = 1
    
    # s2 = A01 * B02
    U[1][8] = 1
    V[2][8] = 1
    
    # s3 = A10 * B12
    U[3][9] = 1
    V[5][9] = 1
    
    # s4 = A11 * B12
    U[4][10] = 1
    V[5][10] = 1
    
    # W contributions for C[:, 2]
    # C20 = s1 + s2
    W[4][7] = 1
    W[4][8] = 1
    
    # C21 = s3 + s4
    W[5][9] = 1
    W[5][10] = 1
    
    # Part 3: Schoolbook for A_R (col 2) * B_bottom (row 2)
    # A_R = [[A02], [A12]]
    # B_bottom = [[B20, B21, B22]]
    
    # b1 = A02 * B20
    U[2][11] = 1
    V[6][11] = 1
    
    # b2 = A02 * B21
    U[2][12] = 1
    V[7][12] = 1
    
    # b3 = A02 * B22
    U[2][13] = 1
    V[8][13] = 1
    
    # b4 = A12 * B20
    U[5][14] = 1
    V[6][14] = 1
    
    # b5 = A12 * B21
    U[5][15] = 1
    V[7][15] = 1
    
    # b6 = A12 * B22
    U[5][16] = 1
    V[8][16] = 1
    
    # W contributions for C
    # C00 += b1
    W[0][11] = 1
    # C01 += b2
    W[1][12] = 1
    # C20 += b3
    W[4][13] = 1
    # C10 += b4
    W[2][14] = 1
    # C11 += b5
    W[3][15] = 1
    # C21 += b6
    W[5][16] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
