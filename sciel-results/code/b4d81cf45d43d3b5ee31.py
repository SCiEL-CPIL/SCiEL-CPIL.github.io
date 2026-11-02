"""Decomposition using Kronecker product of Strassen for 4x4x4 core, extended for 4x4x7."""
DIMENSIONS = [4, 4, 7]

def construct():
    """
    Decomposition: split 4x4x7 into 4x4x4 (Kronecker of Strassen, rank 49) 
    and 4x4x3 (schoolbook, rank 48). Total rank: 97.
    """
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Strassen 2x2x2 decomposition
    U2 = [
        [1, 1, 1, 0, 1, 0, 0],   # a00: row 0
        [0, 0, 0, 1, 0, 0, -1],  # a01: row 1
        [0, 1, 0, 0, 0, 1, 0],   # a10: row 2
        [1, 1, 0, 1, 1, 0, 0],   # a11: row 3
    ]
    V2 = [
        [1, 1, 0, 1, 0, 1, 0],   # b00: row 0
        [0, 0, 1, 0, 0, 0, 1],   # b01: row 1
        [1, 0, 1, 1, 1, 0, 1],   # b10: row 2
        [1, 0, 0, 0, 1, 0, 0],   # b11: row 3
    ]
    W2 = [
        [1, 0, 0, 0, 1, 0, 0],   # c00: row 0
        [0, 0, 1, 0, 0, 0, 1],   # c01: row 1
        [0, 1, 0, 1, 0, 1, 0],   # c10: row 2
        [1, -1, 1, 0, -1, 0, 0], # c11: row 3
    ]
    
    # Part 1: 4x4x4 via Kronecker product of Strassen
    rank44 = 49
    U1 = [[0]*rank44 for _ in range(16)]
    V1 = [[0]*rank44 for _ in range(16)]
    W1 = [[0]*rank44 for _ in range(16)]
    
    col = 0
    for r in range(7):
        for s in range(7):
            for i1 in range(2):
                for i0 in range(2):
                    for j1 in range(2):
                        for j0 in range(2):
                            i = 2*i1 + i0
                            j = 2*j1 + j0
                            U1[i*4 + j][col] = float(U2[i1*2 + j1][r] * U2[i0*2 + j0][s])
                        
                        for k1 in range(2):
                            for k0 in range(2):
                                k = 2*k1 + k0
                                V1[j*4 + k][col] = float(V2[j1*2 + k1][r] * V2[j0*2 + k0][s])
                                W1[k*4 + i][col] = float(W2[k1*2 + i1][r] * W2[k0*2 + i0][s])
            col += 1
    
    # Part 2: 4x4x3 schoolbook
    rank43 = 4 * 4 * 3
    U2x = [[0]*rank43 for _ in range(16)]
    V2x = [[0]*rank43 for _ in range(12)]
    W2x = [[0]*rank43 for _ in range(12)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U2x[i*4 + j][col] = 1
                V2x[j*3 + k][col] = 1
                W2x[k*4 + i][col] = 1
                col += 1
    
    total_rank = rank44 + rank43
    
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(28)]
    W = [[0]*total_rank for _ in range(28)]
    
    for row in range(16):
        for c in range(rank44):
            U[row][c] = U1[row][c]
        for c in range(rank43):
            U[row][rank44 + c] = U2x[row][c]
    
    for j in range(4):
        for k in range(4):
            v1_row = j*4 + k
            v_row = j*7 + k
            for c in range(rank44):
                V[v_row][c] = V1[v1_row][c]
    
    for j in range(4):
        for k_rel in range(3):
            k_full = 4 + k_rel
            v2x_row = j*3 + k_rel
            v_row = j*7 + k_full
            for c in range(rank43):
                V[v_row][rank44 + c] = V2x[v2x_row][c]
    
    for k in range(4):
        for i in range(4):
            w_row = k*4 + i
            for c in range(rank44):
                W[w_row][c] = W1[w_row][c]
    
    for k_rel in range(3):
        k_full = 4 + k_rel
        for i in range(4):
            w_row = k_full*4 + i
            for c in range(rank43):
                W[w_row][rank44 + c] = W2x[k_rel*4 + i][c]
    
    return {"U": U, "V": V, "W": W}
