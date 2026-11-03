"""Matrix multiplication 2x3 by 3x3 - rank 17 decomposition via Strassen + trivial blocks."""

def construct():
    n, m, p = 2, 3, 3
    # We decompose into 4 blocks:
    # A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    # B = [[B11, B12],
    #      [B21, B22]] where B11 is 2x2, B12 is 2x1, B21 is 1x2, B22 is 1x1
    # C = A1*B11 + A2*B21 (for cols 1,2) and A1*B12 + A2*B22 (for col 3)
    #
    # Block 1: A1 (2x2) * B11 (2x2) -> use Strassen (rank 7)
    # Block 2: A2 (2x1) * B21 (1x2) -> trivial (rank 4)
    # Block 3: A1 (2x2) * B12 (2x1) -> trivial (rank 4)
    # Block 4: A2 (2x1) * B22 (1x1) -> trivial (rank 2)
    # Total rank = 7 + 4 + 4 + 2 = 17
    
    rank = 17
    U = [[0] * rank for _ in range(n * m)]  # 6 x 17
    V = [[0] * rank for _ in range(m * p)]  # 9 x 17
    W = [[0] * rank for _ in range(p * n)]  # 6 x 17
    
    # Indexing:
    # A[i,j] -> U[i*m + j] = U[i*3 + j]
    # B[j,k] -> V[j*p + k] = V[j*3 + k]
    # C[i,k] -> W[k*n + i] = W[k*2 + i]
    
    # A1 = A[0:2, 0:2] -> indices in A: (0,0),(0,1),(1,0),(1,1) -> U indices 0,1,3,4
    # A2 = A[0:2, 2]   -> indices in A: (0,2),(1,2) -> U indices 2,5
    # B11 = B[0:2, 0:2] -> indices in B: (0,0),(0,1),(1,0),(1,1) -> V indices 0,1,3,4
    # B12 = B[0:2, 2]   -> indices in B: (0,2),(1,2) -> V indices 2,5
    # B21 = B[2, 0:2]   -> indices in B: (2,0),(2,1) -> V indices 6,7
    # B22 = B[2, 2]     -> index in B: (2,2) -> V index 8
    
    # Strassen for A1 * B11 (2x2 * 2x2):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For A1: A00=U[0], A01=U[1], A10=U[3], A11=U[4]
    # For B11: B00=V[0], B01=V[1], B10=V[3], B11=V[4]
    # For C block (cols 0,1): C00=W[0*2+0]=W[0], C01=W[1*2+0]=W[2], C10=W[0*2+1]=W[1], C11=W[1*2+1]=W[3]
    
    # Strassen terms (rank 7), columns 0-6:
    # M1: (A00+A11)(B00+B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[2][0] = 1; W[1][0] = -1; W[3][0] = 1
    
    # M2: (A10+A11)B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3: A00(B10-B11)
    U[0][2] = 1
    V[3][2] = 1; V[4][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4: A11(B01-B10)
    U[4][3] = 1
    V[1][3] = 1; V[3][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5: (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6: (A10-A00)(B00+B10)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[3][5] = 1
    W[3][5] = 1
    
    # M7: (A01-A11)(B01+B11)
    U[1][6] = 1; U[4][6] = -1
    V[1][6] = 1; V[4][6] = 1
    W[0][6] = 1; W[2][6] = -1
    
    # Block 2: A2 (2x1) * B21 (1x2) -> rank 4, columns 7-10
    # A2 = [A02, A12] = [U[2], U[5]]
    # B21 = [B20, B21] = [V[6], V[7]]
    # Result: C02_block = A02*B20 (W[0]), C03_block = A02*B21 (W[1]), C12_block = A12*B20 (W[2]), C13_block = A12*B21 (W[3])
    # Wait, the output C is 2x3. C[i,k] where i=0,1 and k=0,1,2.
    # C[0,0] and C[0,1] come from A1*B11 (cols 0,1). C[0,2] comes from A1*B12 + A2*B22.
    # Similarly C[1,0], C[1,1] from A1*B11. C[1,2] from A1*B12 + A2*B22.
    
    # So for the "left" part (cols 0,1), the contributions are:
    # From A1*B11: handled by Strassen above
    # From A2*B21: A2 is 2x1, B21 is 1x2. Product is 2x2.
    #   C[0,0] += A02*B20, C[0,1] += A02*B21
    #   C[1,0] += A12*B20, C[1,1] += A12*B21
    
    # Column 7: A02 * B20
    U[2][7] = 1
    V[6][7] = 1
    W[0][7] = 1
    
    # Column 8: A02 * B21
    U[2][8] = 1
    V[7][8] = 1
    W[1][8] = 1
    
    # Column 9: A12 * B20
    U[5][9] = 1
    V[6][9] = 1
    W[2][9] = 1
    
    # Column 10: A12 * B21
    U[5][10] = 1
    V[7][10] = 1
    W[3][10] = 1
    
    # Block 3: A1 (2x2) * B12 (2x1) -> rank 4, columns 11-14
    # B12 = [B02, B12] = [V[2], V[5]]
    # Result: C[0,2] += A00*B02 + A01*B12, C[1,2] += A10*B02 + A11*B12
    # W index for C[0,2] = W[2*2+0] = W[4], C[1,2] = W[2*2+1] = W[5]
    
    # Column 11: A00 * B02
    U[0][11] = 1
    V[2][11] = 1
    W[4][11] = 1
    
    # Column 12: A01 * B12
    U[1][12] = 1
    V[5][12] = 1
    W[4][12] = 1
    
    # Column 13: A10 * B02
    U[3][13] = 1
    V[2][13] = 1
    W[5][13] = 1
    
    # Column 14: A11 * B12
    U[4][14] = 1
    V[5][14] = 1
    W[5][14] = 1
    
    # Block 4: A2 (2x1) * B22 (1x1) -> rank 2, columns 15-16
    # B22 = V[8]
    # C[0,2] += A02*B22, C[1,2] += A12*B22
    
    # Column 15: A02 * B22
    U[2][15] = 1
    V[8][15] = 1
    W[4][15] = 1
    
    # Column 16: A12 * B22
    U[5][16] = 1
    V[8][16] = 1
    W[5][16] = 1
    
    return {"U": U, "V": V, "W": W}
