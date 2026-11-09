"""Program to construct a rank-49 tensor decomposition for 4x4 matrix multiplication."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition for 2x2 matrix multiplication (rank 7)
    # U: A matrix, V: B matrix, W: C matrix (with trace convention W[k*n+i, r])
    n, m, p = 2, 2, 2
    rank_2 = 7
    
    # U2[i, r] for A elements: i = i1*m + i2 = i1*2 + i2
    # V2[j, r] for B elements: j = j1*p + j2 = j1*2 + j2
    # W2[k, r] for C elements: k = k1*n + k2 = k1*2 + k2 (trace convention: W[k*n+i, r] where i is row index)
    # In standard matrix mult C = A*B, C[k1*n + k2, r] corresponds to element C[k1, k2]
    
    # Strassen's 7 products:
    # P0 = A00 * (B00 + B11)
    # P1 = (A10 + A11) * B00
    # P2 = (A00 + A01) * B11
    # P3 = A11 * (B10 - B00)
    # P4 = (A00 + A10) * (B01 - B11)
    # P5 = (A01 + A11) * (B00 + B10)
    # P6 = (A00 - A10) * (B01 + B11)
    
    U2 = [[0] * rank_2 for _ in range(4)]  # 4 rows (i=0,1,2,3 corresponding to A00,A01,A10,A11)
    V2 = [[0] * rank_2 for _ in range(4)]  # 4 rows (j=0,1,2,3 corresponding to B00,B01,B10,B11)
    W2 = [[0] * rank_2 for _ in range(4)]  # 4 rows (k=0,1,2,3 corresponding to C00,C01,C10,C11)
    
    # Column 0: P0 = A00 * (B00 + B11)
    U2[0][0] = 1    # A00
    V2[0][0] = 1    # B00
    V2[3][0] = 1    # B11
    W2[0][0] = 1    # C00
    
    # Column 1: P1 = (A10 + A11) * B00
    U2[2][1] = 1    # A10
    U2[3][1] = 1    # A11
    V2[0][1] = 1    # B00
    W2[2][1] = 1    # C10
    
    # Column 2: P2 = (A00 + A01) * B11
    U2[0][2] = 1    # A00
    U2[1][2] = 1    # A01
    V2[3][2] = 1    # B11
    W2[1][2] = 1    # C01
    
    # Column 3: P3 = A11 * (B10 - B00)
    U2[3][3] = 1    # A11
    V2[2][3] = 1    # B10
    V2[0][3] = -1   # B00
    W2[3][3] = 1    # C11
    
    # Column 4: P4 = (A00 + A10) * (B01 - B11)
    U2[0][4] = 1    # A00
    U2[2][4] = 1    # A10
    V2[1][4] = 1    # B01
    V2[3][4] = -1   # B11
    W2[1][4] = 1    # C01
    
    # Column 5: P5 = (A01 + A11) * (B00 + B10)
    U2[1][5] = 1    # A01
    U2[3][5] = 1    # A11
    V2[0][5] = 1    # B00
    V2[2][5] = 1    # B10
    W2[3][5] = 1    # C11
    
    # Column 6: P6 = (A00 - A10) * (B01 + B11)
    U2[0][6] = 1    # A00
    U2[2][6] = -1   # A10
    V2[1][6] = 1    # B01
    V2[3][6] = 1    # B11
    W2[1][6] = -1   # C01 (with negative sign)
    
    # Now construct 4x4 decomposition using Kronecker product
    n, m, p = 4, 4, 4
    rank = rank_2 * rank_2  # 49
    
    U = [[0] * rank for _ in range(n * m)]  # 16 rows
    V = [[0] * rank for _ in range(m * p)]  # 16 rows
    W = [[0] * rank for _ in range(p * n)]  # 16 rows (trace convention: W[k*n+i, r])
    
    for i in range(n * m):
        i1 = i // m  # 0 or 1
        i2 = i % m   # 0 or 1
        for r in range(rank):
            r1 = r // rank_2
            r2 = r % rank_2
            val = U2[i1 * m + i2][r1] * U2[i1 * m + i2][r2]
            if val != 0:
                U[i][r] = val
    
    for j in range(m * p):
        j1 = j // p  # 0 or 1
        j2 = j % p   # 0 or 1
        for r in range(rank):
            r1 = r // rank_2
            r2 = r % rank_2
            val = V2[j1 * p + j2][r1] * V2[j1 * p + j2][r2]
            if val != 0:
                V[j][r] = val
    
    # W uses trace convention: W[k*n+i, r] where i is row index of C
    for k in range(p * n):
        k_row = k // n  # row index of C (i in trace convention)
        k_col = k % n   # column index of C
        for r in range(rank):
            r1 = r // rank_2
            r2 = r % rank_2
            # C[k_row, k_col] gets contribution from W2 at appropriate positions
            # In 2x2, W2[k_idx, r] where k_idx corresponds to C element
            # For 4x4, C[k_row, k_col] = sum over r1,r2 of W2[k_row%2 * 2 + k_col%2, r1] * W2[k_row%2 * 2 + k_col%2, r2]
            # But we need to be careful with the Kronecker structure
            k1 = k_row  # 0 or 1
            k2 = k_col  # 0 or 1
            val = W2[k1 * n + k2][r1] * W2[k1 * n + k2][r2]
            if val != 0:
                W[k][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
