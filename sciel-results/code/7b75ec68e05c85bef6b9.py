"""Matrix multiplication 6x6x6 using tensor product of Strassen and schoolbook."""

def _strassen_2x2():
    """Return U, V, W for 2x2 matrix multiplication via Strassen (rank 7).
    U has shape (4, 7) indexing (i,j) -> 2i+j
    V has shape (4, 7) indexing (j,k) -> 2j+k
    W has shape (4, 7) indexing (k,i) -> 2k+i  (trace convention)
    """
    # Strassen 7 multiplications:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # U[i,j] contributes to input A_{i,j}
    # V[j,k] contributes to input B_{j,k}
    # W[k,i] contributes to output C_{i,k}
    #
    # Indexing: U row = 2*i + j, V row = 2*j + k, W row = 2*k + i
    #
    # Let's build column by column (each Strassen factor)
    # For each factor f, U[:,f], V[:,f], W[:,f]
    
    # M1: (A11+A22)*(B11+B22)
    # U: A11 gets +1, A22 gets +1  -> rows 0 (i=0,j=0) and 3 (i=1,j=1)
    # V: B11 gets +1, B22 gets +1  -> rows 0 (j=0,k=0) and 3 (j=1,k=1)
    # W: C11 += M1, C22 += M1
    #   C11 is i=0,k=0 -> W row = 2*0+0 = 0
    #   C22 is i=1,k=1 -> W row = 2*1+1 = 3
    
    # M2: (A21+A22)*B11
    # U: A21 row=2*1+0=2, A22 row=3
    # V: B11 row=0
    # W: C21 += M2, C22 -= M2
    #   C21 is i=1,k=0 -> W row = 2*0+1 = 1
    #   C22 is i=1,k=1 -> W row = 3
    
    # M3: A11*(B12-B22)
    # U: A11 row=0
    # V: B12 row=1 (j=0,k=1), B22 row=3
    # W: C12 += M3, C22 += M3
    #   C12 is i=0,k=1 -> W row = 2*1+0 = 2
    #   C22 is i=1,k=1 -> W row = 3
    
    # M4: A22*(B21-B11)
    # U: A22 row=3
    # V: B21 row=2 (j=1,k=0), B11 row=0
    # W: C11 += M4, C21 += M4
    #   C11 is i=0,k=0 -> W row = 0
    #   C21 is i=1,k=0 -> W row = 1
    
    # M5: (A11+A12)*B22
    # U: A11 row=0, A12 row=1 (i=0,j=1)
    # V: B22 row=3
    # W: C11 -= M5, C12 += M5
    #   C11 is i=0,k=0 -> W row = 0
    #   C12 is i=0,k=1 -> W row = 2
    
    # M6: (A21-A11)*(B11+B12)
    # U: A21 row=2, A11 row=0
    # V: B11 row=0, B12 row=1
    # W: C22 += M6
    #   C22 is i=1,k=1 -> W row = 3
    
    # M7: (A12-A22)*(B21+B22)
    # U: A12 row=1, A22 row=3
    # V: B21 row=2, B22 row=3
    # W: C11 += M7
    #   C11 is i=0,k=0 -> W row = 0
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: col 0
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2: col 1
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[1][1] += 1; W[3][1] -= 1
    
    # M3: col 2
    U[0][2] += 1
    V[1][2] += 1; V[3][2] -= 1
    W[2][2] += 1; W[3][2] += 1
    
    # M4: col 3
    U[3][3] += 1
    V[2][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[1][3] += 1
    
    # M5: col 4
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1; W[2][4] += 1
    
    # M6: col 5
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # M7: col 6
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1
    
    return U, V, W


def _schoolbook_3x3():
    """Return U, V, W for 3x3 matrix multiplication via schoolbook (rank 27).
    U has shape (9, 27) indexing (i,j) -> 3i+j
    V has shape (9, 27) indexing (j,k) -> 3j+k
    W has shape (9, 27) indexing (k,i) -> 3k+i  (trace convention)
    """
    n = 3
    rank = n * n * n  # 27
    U = [[0]*rank for _ in range(n*n)]
    V = [[0]*rank for _ in range(n*n)]
    W = [[0]*rank for _ in range(n*n)]
    
    col = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i*n + j][col] = 1
                V[j*n + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W


def construct():
    """Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for 6x6 matrix multiplication using the tensor product of Strassen (2x2, rank 7)
    and schoolbook (3x3, rank 27), giving total rank 7*27 = 189.
    
    The 6x6 matrix is viewed as a 2x2 block matrix where each block is 3x3.
    The block-level multiplication uses Strassen, and each block multiplication
    uses schoolbook. The tensor product structure ensures exactness.
    """
    U2, V2, W2 = _strassen_2x2()
    U3, V3, W3 = _schoolbook_3x3()
    
    n2, m2, p2 = 2, 2, 2  # block sizes
    n3, m3, p3 = 3, 3, 3  # local sizes within blocks
    
    rank2 = 7
    rank3 = 27
    total_rank = rank2 * rank3  # 189
    
    n = n2 * n3  # 6
    m = m2 * m3  # 6
    p = p2 * p3  # 6
    
    # U has shape (n*m, total_rank) = (36, 189)
    # Row index: (I, i3, J, j3) -> I*m3*J... let's define carefully
    # Original A is 6x6. Viewed as 2x2 blocks of 3x3.
    # Element A[row, col] where row = I*3 + i3, col = J*3 + j3
    # U row index = row * m + col = (I*3 + i3) * 6 + (J*3 + j3)
    # 
    # In the tensor product:
    # U_total[row, col] where col = s*27 + t
    #   = U2[I*2 + J, s] * U3[i3*3 + j3, t]
    #   where I,J in {0,1} are block indices, i3,j3 in {0,1,2} are local indices
    #   and row = (I*3 + i3)*6 + (J*3 + j3)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    for I in range(n2):
        for i3 in range(n3):
            for J in range(m2):
                for j3 in range(m3):
                    row_u = (I * n3 + i3) * m + (J * m3 + j3)
                    # U3 local index
                    u3_idx = i3 * m3 + j3
                    for s in range(rank2):
                        val_u2 = U2[I * m2 + J][s]
                        if val_u2 == 0:
                            continue
                        for t in range(rank3):
                            val_u3 = U3[u3_idx][t]
                            if val_u3 == 0:
                                continue
                            col = s * rank3 + t
                            U[row_u][col] = val_u2 * val_u3
    
    # V has shape (m*p, total_rank) = (36, 189)
    # Element B[row, col] where row = J*3 + j3, col = K*3 + k3
    # V row index = row * p + col = (J*3 + j3) * 6 + (K*3 + k3)
    # V_total[row, col] = V2[J*2 + K, s] * V3[j3*3 + k3, t]
    
    for J in range(m2):
        for j3 in range(m3):
            for K in range(p2):
                for k3 in range(p3):
                    row_v = (J * m3 + j3) * p + (K * p3 + k3)
                    v3_idx = j3 * p3 + k3
                    for s in range(rank2):
                        val_v2 = V2[J * p2 + K][s]
                        if val_v2 == 0:
                            continue
                        for t in range(rank3):
                            val_v3 = V3[v3_idx][t]
                            if val_v3 == 0:
                                continue
                            col = s * rank3 + t
                            V[row_v][col] = val_v2 * val_v3
    
    # W has shape (p*n, total_rank) = (36, 189)
    # Output C[i,k] where i = I*3 + i3, k = K*3 + k3
    # W row index = k * n + i = (K*3 + k3) * 6 + (I*3 + i3)
    # W_total[row, col] = W2[K*2 + I, s] * W3[k3*3 + i3, t]
    
    for K in range(p2):
        for k3 in range(p3):
            for I in range(n2):
                for i3 in range(n3):
                    row_w = (K * p3 + k3) * n + (I * n3 + i3)
                    w3_idx = k3 * n3 + i3
                    for s in range(rank2):
                        val_w2 = W2[K * n2 + I][s]
                        if val_w2 == 0:
                            continue
                        for t in range(rank3):
                            val_w3 = W3[w3_idx][t]
                            if val_w3 == 0:
                                continue
                            col = s * rank3 + t
                            W[row_w][col] = val_w2 * val_w3
    
    return {"U": U, "V": V, "W": W}
