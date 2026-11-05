"""
Rank-189 exact tensor decomposition for 6x6 matrix multiplication.

Uses the Kronecker product structure: M_6 <= M_2 x M_3.
- M_2 uses Strassen's rank-7 decomposition.
- M_3 uses the schoolbook rank-27 decomposition.
- Total rank = 7 * 27 = 189.

The factor matrices U, V, W have shapes (36, 189), (36, 189), (36, 189) respectively.
Entries are integers.
"""

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns {'U', 'V', 'W'} where:
      U: (n*m, rank) = (36, 189)
      V: (m*p, rank) = (36, 189)
      W: (p*n, rank) = (36, 189)
    
    The tensor decomposition satisfies:
      T[i, j, k, l, r, s] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+l, c]
    which corresponds to (A*B)[i, l] = sum_j A[i, j] * B[j, l]
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # --- Strassen rank-7 decomposition for 2x2 ---
    # Indices: for 2x2, A is 2x2, B is 2x2, C is 2x2
    # A indices: (i, j) where i in {0,1}, j in {0,1}
    # B indices: (j, k) where j in {0,1}, k in {0,1}
    # C indices: (i, k) where i in {0,1}, k in {0,1}
    
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U2: shape (4, 7) - maps A entries to products
    # Row index = i*2 + j, Column = product index (0-6)
    U2 = [[0]*7 for _ in range(4)]
    # A00 -> M1, M3, M5, M6
    U2[0][0] = 1  # M1
    U2[0][2] = 1  # M3
    U2[0][4] = 1  # M5
    U2[0][5] = -1 # M6
    # A01 -> M5, M7
    U2[1][4] = 1  # M5
    U2[1][6] = 1  # M7
    # A10 -> M2, M6
    U2[2][1] = 1  # M2
    U2[2][5] = 1  # M6
    # A11 -> M1, M2, M4, M7
    U2[3][0] = 1  # M1
    U2[3][1] = 1  # M2
    U2[3][3] = 1  # M4
    U2[3][6] = -1 # M7
    
    # V2: shape (4, 7) - maps B entries to products
    # Row index = j*2 + k
    V2 = [[0]*7 for _ in range(4)]
    # B00 -> M1, M2, M4, M6
    V2[0][0] = 1  # M1
    V2[0][1] = 1  # M2
    V2[0][3] = -1 # M4
    V2[0][5] = 1  # M6
    # B01 -> M3, M5, M6
    V2[1][2] = 1  # M3
    V2[1][4] = 1  # M5
    V2[1][5] = 1  # M6
    # B10 -> M4, M7
    V2[2][3] = 1  # M4
    V2[2][6] = 1  # M7
    # B11 -> M1, M3, M5, M7
    V2[3][0] = 1  # M1
    V2[3][2] = -1 # M3
    V2[3][4] = 1  # M5
    V2[3][6] = 1  # M7
    
    # W2: shape (4, 7) - maps products to C entries
    # Row index = i*2 + k
    W2 = [[0]*7 for _ in range(4)]
    # C00 = M1 + M4 - M5 + M7
    W2[0][0] = 1  # M1
    W2[0][3] = 1  # M4
    W2[0][4] = -1 # M5
    W2[0][6] = 1  # M7
    # C01 = M3 + M5
    W2[1][2] = 1  # M3
    W2[1][4] = 1  # M5
    # C10 = M2 + M4
    W2[2][1] = 1  # M2
    W2[2][3] = 1  # M4
    # C11 = M1 - M2 + M3 + M6
    W2[3][0] = 1  # M1
    W2[3][1] = -1 # M2
    W2[3][2] = 1  # M3
    W2[3][5] = 1  # M6
    
    # --- Schoolbook rank-27 decomposition for 3x3 ---
    # U3: shape (9, 27), V3: shape (9, 27), W3: shape (9, 27)
    # For each (i,j,k), one column with U3[i*3+j]=1, V3[j*3+k]=1, W3[k*3+i]=1
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j][col] = 1
                V3[j*3+k][col] = 1
                W3[k*3+i][col] = 1
                col += 1
    
    # --- Kronecker product to get 6x6 decomposition ---
    # The 6x6 matrix is partitioned into 3x3 blocks of 2x2 submatrices.
    # A[i, j] where i = i3*2 + i2, j = j3*2 + j2
    # B[j, k] where j = j3*2 + j2, k = k3*2 + k2
    # C[i, k] where i = i3*2 + i2, k = k3*2 + k2
    #
    # C[i3*2+i2, k3*2+k2] = sum_{j3, j2} A[i3*2+i2, j3*2+j2] * B[j3*2+j2, k3*2+k2]
    #
    # The Kronecker product structure:
    # U6[row, col] = U2[i2*2+j2, c2] * U3[i3*3+j3, c3]
    # V6[row, col] = V2[j2*2+k2, c2] * V3[j3*3+k3, c3]
    # W6[row, col] = W2[i2*2+k2, c2] * W3[i3*3+k3, c3]
    # where col = c2*27 + c3
    #
    # Row for U6: i*6 + j = (i3*2+i2)*6 + (j3*2+j2) = i3*12 + i2*6 + j3*2 + j2
    # Row for V6: j*6 + k = (j3*2+j2)*6 + (k3*2+k2) = j3*12 + j2*6 + k3*2 + k2
    # Row for W6: k*6 + i = (k3*2+k2)*6 + (i3*2+i2) = k3*12 + k2*6 + i3*2 + i2
    
    rank = 7 * 27  # 189
    
    U6 = [[0]*rank for _ in range(36)]
    V6 = [[0]*rank for _ in range(36)]
    W6 = [[0]*rank for _ in range(36)]
    
    for c2 in range(7):
        for c3 in range(27):
            col = c2 * 27 + c3
            for i3 in range(3):
                for j3 in range(3):
                    u3_val = U3[i3*3+j3][c3]
                    for i2 in range(2):
                        for j2 in range(2):
                            u2_val = U2[i2*2+j2][c2]
                            val = u2_val * u3_val
                            if val != 0:
                                row_u = i3*12 + i2*6 + j3*2 + j2
                                U6[row_u][col] += val
            for j3 in range(3):
                for k3 in range(3):
                    v3_val = V3[j3*3+k3][c3]
                    for j2 in range(2):
                        for k2 in range(2):
                            v2_val = V2[j2*2+k2][c2]
                            val = v2_val * v3_val
                            if val != 0:
                                row_v = j3*12 + j2*6 + k3*2 + k2
                                V6[row_v][col] += val
            for i3 in range(3):
                for k3 in range(3):
                    w3_val = W3[i3*3+k3][c3]
                    for i2 in range(2):
                        for k2 in range(2):
                            w2_val = W2[i2*2+k2][c2]
                            val = w2_val * w3_val
                            if val != 0:
                                row_w = k3*12 + k2*6 + i3*2 + i2
                                W6[row_w][col] += val
    
    return {"U": U6, "V": V6, "W": W6}
# EVOLVE-BLOCK-END
