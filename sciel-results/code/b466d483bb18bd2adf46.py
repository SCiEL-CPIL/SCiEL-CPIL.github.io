"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Rank-98 decomposition via Kronecker product of Strassen(2x2x2, rank 7) 
    # and T_{2,2,4} (direct sum of two Strassen blocks, rank 14)
    
    # Strassen 2x2x2 factors
    U1 = [
        [1, 0, 1, 0, 1, -1, 0],   # A11
        [0, 0, 0, 0, 1, 0, -1],   # A12
        [0, 1, 0, 0, 0, 1, 0],    # A21
        [1, 1, 0, 1, 0, 0, -1]    # A22
    ]
    V1 = [
        [1, 1, 0, -1, 0, 1, 0],   # B11
        [0, 0, 1, 0, 0, 1, 0],    # B12
        [0, 0, 0, 1, 0, 0, 1],    # B21
        [1, 0, -1, 0, 1, 0, 1]    # B22
    ]
    W1 = [
        [1, 0, 0, 1, -1, 0, 1],   # C11 (k=0,i=0)
        [0, 0, 1, 0, 1, 0, 0],    # C12 (k=0,i=1)
        [0, 1, 0, 1, 0, 0, 0],    # C21 (k=1,i=0)
        [1, -1, 1, 0, 0, 1, 0]    # C22 (k=1,i=1)
    ]
    
    # T_{2,2,4} = T_{2,2,2} ⊕ T_{2,2,2}: rank 14
    # U2: 4x14, V2: 8x14, W2: 8x14
    U2 = []
    V2 = []
    W2 = []
    
    # Block 1: first 2 columns of the 2x4 B matrix
    for row in U1:
        U2.append(row + [0]*7)
    for row in V1:
        V2.append(row + [0]*7)
    for row in W1:
        W2.append(row + [0]*7)
    
    # Block 2: last 2 columns of the 2x4 B matrix
    for row in U1:
        U2.append([0]*7 + row)
    for row in V1:
        V2.append([0]*7 + row)
    for row in W1:
        W2.append([0]*7 + row)
    
    # Kronecker product parameters
    n1, m1, p1 = 2, 2, 2
    n2, m2, p2 = 2, 2, 4
    rank1, rank2 = 7, 14
    total_rank = rank1 * rank2  # 98
    
    n = n1 * n2  # 4
    m = m1 * m2  # 4
    p = p1 * p2  # 8
    
    # U: (n*m) x total_rank = 16 x 98
    U = [[0] * total_rank for _ in range(n * m)]
    for i1 in range(n1):
        for i2 in range(n2):
            for j1 in range(m1):
                for j2 in range(m2):
                    i = i1 * n2 + i2
                    j = j1 * m2 + j2
                    row_idx = i * m + j
                    for c1 in range(rank1):
                        for c2 in range(rank2):
                            val = U1[i1 * m1 + j1][c1] * U2[i2 * m2 + j2][c2]
                            if val != 0:
                                U[row_idx][c1 * rank2 + c2] = val
    
    # V: (m*p) x total_rank = 32 x 98
    V = [[0] * total_rank for _ in range(m * p)]
    for j1 in range(m1):
        for j2 in range(m2):
            for k1 in range(p1):
                for k2 in range(p2):
                    j = j1 * m2 + j2
                    k = k1 * p2 + k2
                    row_idx = j * p + k
                    for c1 in range(rank1):
                        for c2 in range(rank2):
                            val = V1[j1 * p1 + k1][c1] * V2[j2 * p2 + k2][c2]
                            if val != 0:
                                V[row_idx][c1 * rank2 + c2] = val
    
    # W: (p*n) x total_rank = 32 x 98
    # W[k*n + i, c]
    W = [[0] * total_rank for _ in range(p * n)]
    for k1 in range(p1):
        for k2 in range(p2):
            for i1 in range(n1):
                for i2 in range(n2):
                    k = k1 * p2 + k2
                    i = i1 * n2 + i2
                    row_idx = k * n + i
                    for c1 in range(rank1):
                        for c2 in range(rank2):
                            val = W1[k1 * n1 + i1][c1] * W2[k2 * n2 + i2][c2]
                            if val != 0:
                                W[row_idx][c1 * rank2 + c2] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
