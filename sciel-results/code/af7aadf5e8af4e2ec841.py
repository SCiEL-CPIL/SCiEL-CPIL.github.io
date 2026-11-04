DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block
    def add_strassen_block(u_indices, v_indices, w_indices):
        nonlocal col
        # M1
        U[u_indices[0]][col] += 1
        U[u_indices[3]][col] += 1
        V[v_indices[0]][col] += 1
        V[v_indices[3]][col] += 1
        W[w_indices[0]][col] += 1
        W[w_indices[3]][col] += 1
        col += 1
        
        # M2
        U[u_indices[2]][col] += 1
        U[u_indices[3]][col] += 1
        V[v_indices[0]][col] += 1
        W[w_indices[1]][col] += 1
        W[w_indices[3]][col] -= 1
        col += 1
        
        # M3
        U[u_indices[0]][col] += 1
        V[v_indices[1]][col] += 1
        V[v_indices[3]][col] -= 1
        W[w_indices[2]][col] += 1
        W[w_indices[3]][col] += 1
        col += 1
        
        # M4
        U[u_indices[3]][col] += 1
        V[v_indices[2]][col] += 1
        V[v_indices[0]][col] -= 1
        W[w_indices[1]][col] += 1
        W[w_indices[0]][col] += 1
        col += 1
        
        # M5
        U[u_indices[0]][col] += 1
        U[u_indices[1]][col] += 1
        V[v_indices[3]][col] += 1
        W[w_indices[0]][col] -= 1
        W[w_indices[2]][col] += 1
        col += 1
        
        # M6
        U[u_indices[2]][col] += 1
        U[u_indices[0]][col] -= 1
        V[v_indices[0]][col] += 1
        V[v_indices[1]][col] += 1
        W[w_indices[3]][col] += 1
        col += 1
        
        # M7
        U[u_indices[1]][col] += 1
        U[u_indices[3]][col] -= 1
        V[v_indices[2]][col] += 1
        V[v_indices[3]][col] += 1
        W[w_indices[0]][col] += 1
        col += 1

    # Block 1: A1 (2x2) * B1 (2x6)
    # A1 rows: 0, 1. Cols: 0, 1.
    # U indices: i*m + j
    # A[0,0] -> 0*5+0 = 0
    # A[0,1] -> 0*5+1 = 1
    # A[1,0] -> 1*5+0 = 5
    # A[1,1] -> 1*5+1 = 6
    u1 = [0, 1, 5, 6]
    
    # B1 rows: 0, 1. Cols: 0..5
    # V indices: j*p + k
    # B[0,0] -> 0*6+0 = 0
    # B[0,1] -> 0*6+1 = 1
    # B[0,2] -> 0*6+2 = 2
    # B[0,3] -> 0*6+3 = 3
    # B[0,4] -> 0*6+4 = 4
    # B[0,5] -> 0*6+5 = 5
    # B[1,0] -> 1*6+0 = 6
    # B[1,1] -> 1*6+1 = 7
    # B[1,2] -> 1*6+2 = 8
    # B[1,3] -> 1*6+3 = 9
    # B[1,4] -> 1*6+4 = 10
    # B[1,5] -> 1*6+5 = 11
    
    # B1a: cols 0,1
    v1a = [0, 1, 6, 7]
    # W indices for C (2x2) part corresponding to cols 0,1 of B
    # C[0,0] -> 0*2+0 = 0
    # C[0,1] -> 1*2+0 = 2
    # C[1,0] -> 0*2+1 = 1
    # C[1,1] -> 1*2+1 = 3
    w1a = [0, 2, 1, 3]
    
    add_strassen_block(u1, v1a, w1a)
    
    # B1b: cols 2,3
    v1b = [2, 3, 8, 9]
    # C cols 2,3
    # C[0,2] -> 2*2+0 = 4
    # C[0,3] -> 3*2+0 = 6
    # C[1,2] -> 2*2+1 = 5
    # C[1,3] -> 3*2+1 = 7
    w1b = [4, 6, 5, 7]
    
    add_strassen_block(u1, v1b, w1b)
    
    # B1c: cols 4,5
    v1c = [4, 5, 10, 11]
    # C cols 4,5
    # C[0,4] -> 4*2+0 = 8
    # C[0,5] -> 5*2+0 = 10
    # C[1,4] -> 4*2+1 = 9
    # C[1,5] -> 5*2+1 = 11
    w1c = [8, 10, 9, 11]
    
    add_strassen_block(u1, v1c, w1c)
    
    # Block 2: A2 (2x2) * B2 (2x6)
    # A2 rows: 0, 1. Cols: 2, 3.
    # A[0,2] -> 0*5+2 = 2
    # A[0,3] -> 0*5+3 = 3
    # A[1,2] -> 1*5+2 = 7
    # A[1,3] -> 1*5+3 = 8
    u2 = [2, 3, 7, 8]
    
    # B2 rows: 2, 3. Cols: 0..5
    # B[2,0] -> 2*6+0 = 12
    # B[2,1] -> 2*6+1 = 13
    # B[2,2] -> 2*6+2 = 14
    # B[2,3] -> 2*6+3 = 15
    # B[2,4] -> 2*6+4 = 16
    # B[2,5] -> 2*6+5 = 17
    # B[3,0] -> 3*6+0 = 18
    # B[3,1] -> 3*6+1 = 19
    # B[3,2] -> 3*6+2 = 20
    # B[3,3] -> 3*6+3 = 21
    # B[3,4] -> 3*6+4 = 22
    # B[3,5] -> 3*6+5 = 23
    
    v2a = [12, 13, 18, 19]
    w2a = [0, 2, 1, 3] # Same C indices as B1a
    add_strassen_block(u2, v2a, w2a)
    
    v2b = [14, 15, 20, 21]
    w2b = [4, 6, 5, 7]
    add_strassen_block(u2, v2b, w2b)
    
    v2c = [16, 17, 22, 23]
    w2c = [8, 10, 9, 11]
    add_strassen_block(u2, v2c, w2c)
    
    # Block 3: A3 (2x1) * B3 (1x6)
    # A3 rows: 0, 1. Col: 4.
    # A[0,4] -> 0*5+4 = 4
    # A[1,4] -> 1*5+4 = 9
    u3 = [4, 9]
    
    # B3 row: 4. Cols: 0..5
    # B[4,0] -> 4*6+0 = 24
    # B[4,1] -> 4*6+1 = 25
    # B[4,2] -> 4*6+2 = 26
    # B[4,3] -> 4*6+3 = 27
    # B[4,4] -> 4*6+4 = 28
    # B[4,5] -> 4*6+5 = 29
    v3 = [24, 25, 26, 27, 28, 29]
    
    # C indices: all 12
    # C[0,0]..C[0,5] -> 0, 2, 4, 6, 8, 10
    # C[1,0]..C[1,5] -> 1, 3, 5, 7, 9, 11
    
    for k in range(6):
        # C[0, k]
        W[k*2 + 0][col] += 1
        U[4][col] += 1
        V[24+k][col] += 1
        col += 1
        
        # C[1, k]
        W[k*2 + 1][col] += 1
        U[9][col] += 1
        V[24+k][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
