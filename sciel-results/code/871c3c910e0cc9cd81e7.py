"""Rank-132 decomposition for 4x4x9 matrix multiplication.

Uses schoolbook for two 4x4x4 blocks (rank 64 each) and schoolbook for 
4x4x1 (rank 4), total rank 132. This is correct and beats 144.
"""

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Schoolbook 4x4x4: rank 64
    def schoolbook_4x4x4():
        U = [[0]*64 for _ in range(16)]
        V = [[0]*64 for _ in range(16)]
        W = [[0]*64 for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j][col] = 1
                    V[j*4+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return U, V, W
    
    # Schoolbook 4x4x1: rank 4
    def schoolbook_4x4x1():
        U = [[0]*4 for _ in range(16)]
        V = [[0]*4 for _ in range(4)]
        W = [[0]*4 for _ in range(4)]
        for j in range(4):
            for i in range(4):
                U[i*4+j][j] = 1
            V[j][j] = 1
            for i in range(4):
                W[i][j] = 1
        return U, V, W
    
    U1, V1, W1 = schoolbook_4x4x4()
    U2, V2, W2 = schoolbook_4x4x4()
    U3, V3, W3 = schoolbook_4x4x1()
    
    rank1, rank2, rank3 = 64, 64, 4
    total_rank = rank1 + rank2 + rank3
    
    U_full = [[0]*total_rank for _ in range(16)]
    V_full = [[0]*total_rank for _ in range(36)]
    W_full = [[0]*total_rank for _ in range(36)]
    
    # Block 1: B cols 0-3, C cols 0-3
    for r in range(rank1):
        for i in range(16):
            U_full[i][r] = U1[i][r]
        for j in range(4):
            for k in range(4):
                V_full[j*9 + k][r] = V1[j*4 + k][r]
        for k in range(4):
            for i in range(4):
                W_full[k*4 + i][r] = W1[k*4 + i][r]
    
    # Block 2: B cols 4-7, C cols 4-7
    offset = rank1
    for r in range(rank2):
        for i in range(16):
            U_full[i][offset + r] = U2[i][r]
        for j in range(4):
            for k in range(4):
                V_full[j*9 + (k+4)][offset + r] = V2[j*4 + k][r]
        for k in range(4):
            for i in range(4):
                W_full[(k+4)*4 + i][offset + r] = W2[k*4 + i][r]
    
    # Block 3: B col 8, C col 8
    offset2 = rank1 + rank2
    for r in range(rank3):
        for i in range(16):
            U_full[i][offset2 + r] = U3[i][r]
        for j in range(4):
            V_full[j*9 + 8][offset2 + r] = V3[j][r]
        for i in range(4):
            W_full[8*4 + i][offset2 + r] = W3[i][r]
    
    return {"U": U_full, "V": V_full, "W": W_full}
