"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    # Rank 23 decomposition for 3x3 matrix multiplication
    n, m, p = DIMENSIONS
    rank = 23
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Using the known rank-23 decomposition for 3x3 matrix multiplication
    # This is based on the work by Landsberg and others showing that 3x3 matrix
    # multiplication can be done with rank 23.
    
    # The decomposition uses the following pattern where we exploit the structure
    # of 3x3 matrices to reduce the rank from 27 (schoolbook) to 23.
    
    # Term 0: A00*B00 -> C00
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    
    # Term 1: A01*B10 -> C00
    U[1][1] = 1; V[3][1] = 1; W[0][1] = 1
    
    # Term 2: A02*B20 -> C00
    U[2][2] = 1; V[6][2] = 1; W[0][2] = 1
    
    # Term 3: A10*B01 -> C11
    U[3][3] = 1; V[1][3] = 1; W[4][3] = 1
    
    # Term 4: A11*B11 -> C11
    U[4][4] = 1; V[4][4] = 1; W[4][4] = 1
    
    # Term 5: A12*B21 -> C11
    U[5][5] = 1; V[7][5] = 1; W[4][5] = 1
    
    # Term 6: A20*B02 -> C22
    U[6][6] = 1; V[2][6] = 1; W[8][6] = 1
    
    # Term 7: A21*B12 -> C22
    U[7][7] = 1; V[5][7] = 1; W[8][7] = 1
    
    # Term 8: A22*B22 -> C22
    U[8][8] = 1; V[8][8] = 1; W[8][8] = 1
    
    # Term 9: A00*B01 -> C01
    U[0][9] = 1; V[1][9] = 1; W[3][9] = 1
    
    # Term 10: A01*B11 -> C01
    U[1][10] = 1; V[4][10] = 1; W[3][10] = 1
    
    # Term 11: A02*B21 -> C01
    U[2][11] = 1; V[7][11] = 1; W[3][11] = 1
    
    # Term 12: A00*B02 -> C02
    U[0][12] = 1; V[2][12] = 1; W[6][12] = 1
    
    # Term 13: A01*B12 -> C02
    U[1][13] = 1; V[5][13] = 1; W[6][13] = 1
    
    # Term 14: A02*B22 -> C02
    U[2][14] = 1; V[8][14] = 1; W[6][14] = 1
    
    # Term 15: A10*B00 -> C10
    U[3][15] = 1; V[0][15] = 1; W[1][15] = 1
    
    # Term 16: A11*B10 -> C10
    U[4][16] = 1; V[3][16] = 1; W[1][16] = 1
    
    # Term 17: A12*B20 -> C10
    U[5][17] = 1; V[6][17] = 1; W[1][17] = 1
    
    # Term 18: A10*B02 -> C12
    U[3][18] = 1; V[2][18] = 1; W[7][18] = 1
    
    # Term 19: A11*B12 -> C12
    U[4][19] = 1; V[5][19] = 1; W[7][19] = 1
    
    # Term 20: A12*B22 -> C12
    U[5][20] = 1; V[8][20] = 1; W[7][20] = 1
    
    # Term 21: A20*B01 -> C21
    U[6][21] = 1; V[1][21] = 1; W[5][21] = 1
    
    # Term 22: A21*B11 -> C21
    U[7][22] = 1; V[4][22] = 1; W[5][22] = 1
    
    # We need 4 more terms to get to rank 23, but we only have 23 terms above.
    # The above is actually rank 23 if all terms are independent.
    # Let me verify the count: terms 0-22 = 23 terms.
    
    # Wait, I only listed 23 terms but some output positions might be missing.
    # Let me check which C[i,k] positions are covered:
    # C00: terms 0,1,2 -> covered
    # C01: terms 9,10,11 -> covered
    # C02: terms 12,13,14 -> covered
    # C10: terms 15,16,17 -> covered
    # C11: terms 3,4,5 -> covered
    # C12: terms 18,19,20 -> covered
    # C20: MISSING!
    # C21: terms 21,22 -> only 2 terms, need 3
    # C22: terms 6,7,8 -> covered
    
    # I'm missing C20 and one term for C21. This means my decomposition is incomplete.
    # The rank 23 decomposition is more complex than just listing terms.
    
    # For safety, let me use the schoolbook rank 27 decomposition which is guaranteed correct.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
