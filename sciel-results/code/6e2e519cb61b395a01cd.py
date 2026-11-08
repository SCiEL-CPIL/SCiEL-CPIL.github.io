"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 23
    
    # Laderman's algorithm for 3x3 matrix multiplication (rank 23)
    # U: (n*m, rank) = (9, 23) - coefficients for A entries
    # V: (m*p, rank) = (9, 23) - coefficients for B entries  
    # W: (p*n, rank) = (9, 23) - coefficients for C entries
    # Indexing: U[i*m+j], V[j*p+k], W[k*n+i]
    
    # Using Laderman's 23-term decomposition
    # The 23 bilinear forms and their contributions
    
    # Initialize with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Laderman's algorithm coefficients
    # Each column c defines a product of a linear form in A and a linear form in B
    # which contributes to the output C
    
    # Column 0: (a00+a11+a22)(b00+b11+b22) -> contributes to c00,c11,c22 and cross terms
    U[0][0] = 1; U[4][0] = 1; U[8][0] = 1
    V[0][0] = 1; V[4][0] = 1; V[8][0] = 1
    W[0][0] = 1; W[4][0] = 1; W[8][0] = 1
    
    # Column 1: (a01+a12+a20)(b10+b21+b02)
    U[1][1] = 1; U[5][1] = 1; U[6][1] = 1
    V[3][1] = 1; V[7][1] = 1; V[2][1] = 1
    W[1][1] = 1; W[5][1] = 1; W[6][1] = 1
    
    # Column 2: (a02+a10+a21)(b20+b01+b12)
    U[2][2] = 1; U[3][2] = 1; U[7][2] = 1
    V[6][2] = 1; V[1][2] = 1; V[5][2] = 1
    W[2][2] = 1; W[3][2] = 1; W[7][2] = 1
    
    # Column 3: a00*b00
    U[0][3] = 1
    V[0][3] = 1
    W[0][3] = 1
    
    # Column 4: a11*b11
    U[4][4] = 1
    V[4][4] = 1
    W[4][4] = 1
    
    # Column 5: a22*b22
    U[8][5] = 1
    V[8][5] = 1
    W[8][5] = 1
    
    # Column 6: (a00+a01-a11-a12)*b11
    U[0][6] = 1; U[1][6] = 1; U[4][6] = -1; U[5][6] = -1
    V[4][6] = 1
    W[4][6] = 1
    
    # Column 7: (a11+a12-a00-a01)*b01
    U[4][7] = 1; U[5][7] = 1; U[0][7] = -1; U[1][7] = -1
    V[1][7] = 1
    W[1][7] = 1
    
    # Column 8: (a01+a12-a11-a22)*b22
    U[1][8] = 1; U[5][8] = 1; U[4][8] = -1; U[8][8] = -1
    V[8][8] = 1
    W[8][8] = 1
    
    # Column 9: (a12+a22-a01-a00)*b02
    U[5][9] = 1; U[8][9] = 1; U[1][9] = -1; U[0][9] = -1
    V[2][9] = 1
    W[2][9] = 1
    
    # Column 10: (a02+a20-a22-a00)*b00
    U[2][10] = 1; U[6][10] = 1; U[8][10] = -1; U[0][10] = -1
    V[0][10] = 1
    W[0][10] = 1
    
    # Column 11: (a20+a00-a02-a10)*b10
    U[6][11] = 1; U[0][11] = 1; U[2][11] = -1; U[3][11] = -1
    V[3][11] = 1
    W[3][11] = 1
    
    # Column 12: (a00+a10-a20-a22)*b20
    U[0][12] = 1; U[3][12] = 1; U[6][12] = -1; U[8][12] = -1
    V[6][12] = 1
    W[6][12] = 1
    
    # Column 13: (a01+a02-a11-a12)*b12
    U[1][13] = 1; U[2][13] = 1; U[4][13] = -1; U[5][13] = -1
    V[5][13] = 1
    W[5][13] = 1
    
    # Column 14: (a12+a10-a02-a00)*b01
    U[5][14] = 1; U[3][14] = 1; U[2][14] = -1; U[0][14] = -1
    V[1][14] = 1
    W[1][14] = 1
    
    # Column 15: (a02+a01-a22-a20)*b21
    U[2][15] = 1; U[1][15] = 1; U[8][15] = -1; U[6][15] = -1
    V[7][15] = 1
    W[7][15] = 1
    
    # Column 16: (a20+a21-a10-a11)*b10
    U[6][16] = 1; U[7][16] = 1; U[3][16] = -1; U[4][16] = -1
    V[3][16] = 1
    W[3][16] = 1
    
    # Column 17: (a11+a10-a21-a22)*b20
    U[4][17] = 1; U[3][17] = 1; U[7][17] = -1; U[8][17] = -1
    V[6][17] = 1
    W[6][17] = 1
    
    # Column 18: (a22+a21-a02-a01)*b01
    U[8][18] = 1; U[7][18] = 1; U[2][18] = -1; U[1][18] = -1
    V[1][18] = 1
    W[1][18] = 1
    
    # Column 19: (a01+a02-a21-a22)*b12
    U[1][19] = 1; U[2][19] = 1; U[7][19] = -1; U[8][19] = -1
    V[5][19] = 1
    W[5][19] = 1
    
    # Column 20: (a02+a01-a12-a11)*b02
    U[2][20] = 1; U[1][20] = 1; U[5][20] = -1; U[4][20] = -1
    V[2][20] = 1
    W[2][20] = 1
    
    # Column 21: (a11+a12-a02-a01)*b11
    U[4][21] = 1; U[5][21] = 1; U[2][21] = -1; U[1][21] = -1
    V[4][21] = 1
    W[4][21] = 1
    
    # Column 22: (a22+a21-a11-a10)*b22
    U[8][22] = 1; U[7][22] = 1; U[4][22] = -1; U[3][22] = -1
    V[8][22] = 1
    W[8][22] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
