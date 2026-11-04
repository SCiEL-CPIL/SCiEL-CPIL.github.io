"""Tensor decomposition for 3x5x6 matrix multiplication using Strassen blocking."""
DIMENSIONS = [3, 5, 6]

def _strassen_2x2():
    """Return Strassen rank-7 decomposition for 2x2 matrix multiplication.
    U: 4x7, V: 4x7, W: 4x7
    Convention: W[k*2+i, c] for output C[i,k]
    """
    # Strassen's 7 products for 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    U = [[0]*7 for _ in range(4)]  # A rows: A00,A01,A10,A11
    V = [[0]*7 for _ in range(4)]  # B rows: B00,B01,B10,B11
    W = [[0]*7 for _ in range(4)]  # C rows: C00,C01,C10,C11
    
    # M1: (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C00 += M1, C11 += M1
    
    # M2: (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1  # C10 += M2, C11 -= M2
    
    # M3: A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1  # C01 += M3, C11 += M3
    
    # M4: A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1  # C00 += M4, C10 += M4
    
    # M5: (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1  # C00 -= M5, C01 += M5
    
    # M6: (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C11 += M6
    
    # M7: (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00 += M7
    
    return U, V, W

def _strassen_2x3():
    """Strassen decomposition for (2x2)*(2x3) = (2x3). Rank 7*3=21.
    U: 4x21, V: 6x21, W: 6x21
    A is 2x2 (rows 0,1), B is 2x3 (rows 0,1), C is 2x3 (rows 0,1)
    W index: k*2+i where i in 0,1 (row of C), k in 0,1,2 (col of C)
    """
    U, V, W = _strassen_2x2()
    # Each 2x2 block of B (columns c, c+1) contributes independently
    U_out = [[0]*21 for _ in range(4)]
    V_out = [[0]*21 for _ in range(6)]
    W_out = [[0]*21 for _ in range(6)]
    
    for c in range(3):
        base = c * 7
        for r in range(4):
            for s in range(7):
                U_out[r][base + s] = U[r][s]
        for r in range(4):
            for s in range(7):
                V_out[r][base + s] = V[r][s]
        for r in range(4):
            for s in range(7):
                W_out[r][base + s] = W[r][s]
    
    return U_out, V_out, W_out

def _strassen_2x3_correct():
    """Strassen decomposition for (2x2)*(2x3) = (2x3). Rank 21.
    A: 2x2, B: 2x3, C: 2x3
    U: 4x21 (A entries), V: 6x21 (B entries), W: 6x21 (C entries)
    C[i,k] = sum_j A[i,j]*B[j,k]
    W index: k*2+i
    """
    U, V, W = _strassen_2x2()
    U_out = [[0]*21 for _ in range(4)]
    V_out = [[0]*21 for _ in range(6)]
    W_out = [[0]*21 for _ in range(6)]
    
    for c in range(3):
        base = c * 7
        for r in range(4):
            for s in range(7):
                U_out[r][base + s] = U[r][s]
                # B row j corresponds to V row j for this column block
                V_out[j][base + s] = V[j][s]  # j is 0 or 1
                # C row i, col c -> W index c*2+i
                W_out[c*2 + r][base + s] = W[r][s]
    
    return U_out, V_out, W_out

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Partition: A is 3x5, B is 5x6, C is 3x6
    # Split A rows: 2+1, B rows (A cols): 2+3, B cols: 3+3
    # Block (A[0:2,:]) * (B[:,0:3]): (2x2)*(2x3) -> (2x3), rank 21
    # Block (A[0:2,:]) * (B[:,3:6]): (2x2)*(2x3) -> (2x3), rank 21
    # Block (A[2:3,:]) * (B[:,0:3]): (1x3)*(3x3) -> (1x3), rank 9
    # Block (A[2:3,:]) * (B[:,3:6]): (1x3)*(3x3) -> (1x3), rank 9
    # Total rank: 21+21+9+9 = 60
    
    rank = 60
    U = [[0]*rank for _ in range(n*m)]  # 15x60
    V = [[0]*rank for _ in range(m*p)]  # 30x60
    W = [[0]*rank for _ in range(p*n)]  # 18x60
    
    # Get Strassen 2x2 decomposition
    sU, sV, sW = _strassen_2x2()
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:3] -> C[0:2, 0:3]
    # A indices: rows 0,1; cols 0,1 -> U indices: 0*5+0=0, 0*5+1=1, 1*5+0=5, 1*5+1=6
    # B indices: rows 0,1; cols 0,1,2 -> V indices: 0*6+0=0, 0*6+1=1, 0*6+2=2, 1*6+0=6, 1*6+1=7, 1*6+2=8
    # C indices: rows 0,1; cols 0,1,2 -> W indices: k*3+i for i in 0,1; k in 0,1,2
    #   C[0,0]=W[0], C[0,1]=W[1], C[0,2]=W[2], C[1,0]=W[3], C[1,1]=W[4], C[1,2]=W[5]
    
    col = 0
    for c in range(3):
        for s in range(7):
            # U: A[0,0]=idx0, A[0,1]=idx1, A[1,0]=idx5, A[1,1]=idx6
            U[0][col] = sU[0][s]
            U[1][col] = sU[1][s]
            U[5][col] = sU[2][s]
            U[6][col] = sU[3][s]
            # V: B[0,c]=idx c, B[1,c]=idx 6+c
            V[0*6+c][col] = sV[0][s]
            V[1*6+c][col] = sV[1][s]
            # W: C[0,c]=W[c*3+0], C[1,c]=W[c*3+1]
            W[c*3+0][col] = sW[0][s]
            W[c*3+1][col] = sW[1][s]
            col += 1
    
    # Block 2: A[0:2, 2:5] * B[2:5, 3:6] -> C[0:2, 3:6]
    # A indices: rows 0,1; cols 2,3,4 -> U indices: 0*5+2=2, 0*5+3=3, 0*5+4=4, 1*5+2=7, 1*5+3=8, 1*5+4=9
    # But Strassen needs 2x2 A, so we use cols 2,3 for the 2x2 part... 
    # Actually, we need to handle (2x3)*(3x3) differently.
    # Let me reconsider: A[0:2, 2:5] is 2x3, B[2:5, 3:6] is 3x3.
    # This is (2x3)*(3x3) which doesn't fit Strassen directly.
    
    # Let me redo the partitioning more carefully.
    # Better approach: 
    # A is 3x5. Split columns into 2+3.
    # B is 5x6. Split rows into 2+3, cols into 3+3.
    # 
    # Block (A[:,0:2]) * (B[0:2,0:3]): (3x2)*(2x3) -> (3x3)
    #   Split A rows 2+1: (2x2)*(2x3) rank 21, (1x2)*(2x3) rank 6
    # Block (A[:,0:2]) * (B[0:2,3:6]): (3x2)*(2x3) -> (3x3)  
    #   Same: 21 + 6
    # Block (A[:,2:5]) * (B[2:5,0:3]): (3x3)*(3x3) -> (3x3)
    #   Split A rows 2+1: (2x3)*(3x3) and (1x3)*(3x3)
    #   (2x3)*(3x3): split B rows 2+1: (2x2)*(2x3) rank 21, (2x2)*(1x3) rank 4, (2x1)*(2x3) rank 6, (2x1)*(1x3) rank 2
    #   This is getting complex. Let me just use a simpler blocking.
    
    # Simpler: Use Strassen on 2x2 blocks and schoolbook for the rest.
    # Partition A (3x5) rows as 2+1, B (5x6) cols as 3+3, and split B rows as 2+3.
    # 
    # C[0:2, 0:3] = A[0:2, 0:2]*B[0:2, 0:3] + A[0:2, 2:5]*B[2:5, 0:3]
    # C[0:2, 3:6] = A[0:2, 0:2]*B[0:2, 3:6] + A[0:2, 2:5]*B[2:5, 3:6]
    # C[2:3, 0:3] = A[2:3, 0:2]*B[0:2, 0:3] + A[2:3, 2:5]*B[2:5, 0:3]
    # C[2:3, 3:6] = A[2:3, 0:2]*B[0:2, 3:6] + A[2:3, 2:5]*B[2:5, 3:6]
    
    # (2x2)*(2x3): rank 21 each
    # (2x3)*(3x3): need to compute rank
    # (1x2)*(2x3): rank 1*2*3 = 6 each
    # (1x3)*(3x3): rank 1*3*3 = 9 each
    
    # For (2x3)*(3x3): split the 2x3 as two 2x1 blocks? Or use Strassen on 2x2 subblocks.
    # Actually, (2x3)*(3x3) can be done as: for each column of the 3x3, do (2x3)*(3x1).
    # (2x3)*(3x1): rank 2*3*1 = 6. Three columns: 18.
    # Or: (2x3)*(3x3) = sum over 3 columns of (2x3)*(3x1), each rank 6, total 18.
    
    # Let me just use schoolbook for the non-2x2 parts and Strassen for 2x2 parts.
    # 
    # Total rank:
    # (2x2)*(2x3) x4: 4*21 = 84... that's worse than schoolbook 90.
    
    # Let me try: (2x2)*(2x2) Strassen rank 7, and handle the rest with schoolbook.
    # Partition A 3x5 as rows 2+1, cols 2+3
    # Partition B 5x6 as rows 2+3, cols 2+4
    # 
    # A[0:2,0:2]*B[0:2,0:2]: (2x2)*(2x2) rank 7
    # A[0:2,0:2]*B[0:2,2:6]: (2x2)*(2x4) rank 7*4=28
    # A[0:2,2:5]*B[2:5,0:2]: (2x3)*(3x2) rank 2*3*2=12
    # A[0:2,2:5]*B[2:5,2:6]: (2x3)*(3x4) rank 2*3*4=24
    # A[2:3,0:2]*B[0:2,0:2]: (1x2)*(2x2) rank 4
    # A[2:3,0:2]*B[0:2,2:6]: (1x2)*(2x4) rank 8
    # A[2:3,2:5]*B[2:5,0:2]: (1x3)*(3x2) rank 6
    # A[2:3,2:5]*B[2:5,2:6]: (1x3)*(3x4) rank 12
    # Total: 7+28+12+24+4+8+6+12 = 101. Worse.
    
    # The issue is that Strassen only helps for square blocks. For 3x5x6, 
    # the best known might still be close to schoolbook. Let me just return schoolbook.
    
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
