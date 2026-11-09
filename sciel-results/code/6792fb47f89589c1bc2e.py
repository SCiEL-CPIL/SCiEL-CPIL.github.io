def construct():
    import time
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 5, 5, 5
    nm, mp, pn = 25, 25, 25
    
    # Build 4x4x4 Strassen decomposition
    U4, V4, W4 = _strassen_4x4x4()
    
    # Total rank
    rank = 49 + 16 + 16 + 16 + 4 + 4 + 4 + 1  # = 110
    
    U = [[0]*rank for _ in range(nm)]
    V = [[0]*rank for _ in range(mp)]
    W = [[0]*rank for _ in range(pn)]
    
    col = 0
    # Block 1: 4x4x4 (i,j,k in 0..3)
    for i in range(4):
        for j in range(4):
            for r in range(49):
                U[i*5+j][col+r] = U4[i*4+j][r]
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V[j*5+k][col+r] = V4[j*4+k][r]
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W[k*5+i][col+r] = W4[k*4+i][r]
    col += 49
    
    # Block 2: i in 0..3, j in 0..3, k=4 (4x4x1)
    for i in range(4):
        for j in range(4):
            U[i*5+j][col] = 1
            V[j*5+4][col] = 1
            W[4*5+i][col] = 1
            col += 1
    # col = 49+16 = 65
    
    # Block 3: i in 0..3, j=4, k in 0..3 (4x1x4)
    for i in range(4):
        for k in range(4):
            U[i*5+4][col] = 1
            V[4*5+k][col] = 1
            W[k*5+i][col] = 1
            col += 1
    # col = 65+16 = 81
    
    # Block 4: i=4, j in 0..3, k in 0..3 (1x4x4)
    for j in range(4):
        for k in range(4):
            U[4*5+j][col] = 1
            V[j*5+k][col] = 1
            W[k*5+4][col] = 1
            col += 1
    # col = 81+16 = 97
    
    # Block 5: i in 0..3, j=4, k=4 (4x1x1)
    for i in range(4):
        U[i*5+4][col] = 1
        V[4*5+4][col] = 1
        W[4*5+i][col] = 1
        col += 1
    # col = 97+4 = 101
    
    # Block 6: i=4, j in 0..3, k=4 (1x4x1)
    for j in range(4):
        U[4*5+j][col] = 1
        V[j*5+4][col] = 1
        W[4*5+4][col] = 1
        col += 1
    # col = 101+4 = 105
    
    # Block 7: i=4, j=4, k in 0..3 (1x1x4)
    for k in range(4):
        U[4*5+4][col] = 1
        V[4*5+k][col] = 1
        W[k*5+4][col] = 1
        col += 1
    # col = 105+4 = 109
    
    # Block 8: i=4, j=4, k=4 (1x1x1)
    U[4*5+4][col] = 1
    V[4*5+4][col] = 1
    W[4*5+4][col] = 1
    col += 1
    # col = 110
    
    assert col == rank, f"col={col}, rank={rank}"
    
    return {"U": U, "V": V, "W": W}
