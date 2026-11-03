def construct():
    n, m, p = 2, 4, 6
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: j in {0, 1}
    # A indices: i*4 + j
    # B indices: j*6 + k
    # C indices: k*2 + i
    
    col_offset = 0
    for i in range(n):
        for j in range(2): # j=0,1
            for k in range(p): # k=0..5
                # U[i*4+j, col] = 1
                # V[j*6+k, col] = 1
                # W[k*2+i, col] = 1
                U[i*4 + j][col_offset] = 1
                V[j*6 + k][col_offset] = 1
                W[k*2 + i][col_offset] = 1
                col_offset += 1
                
    # Block 2: j in {2, 3}
    for i in range(n):
        for j in range(2, 4): # j=2,3
            for k in range(p): # k=0..5
                # U[i*4+j, col] = 1
                # V[j*6+k, col] = 1
                # W[k*2+i, col] = 1
                U[i*4 + j][col_offset] = 1
                V[j*6 + k][col_offset] = 1
                W[k*2 + i][col_offset] = 1
                col_offset += 1
                
    return {"U": U, "V": V, "W": W}
