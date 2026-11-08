def add_strassen_block(U, V, W, col_start, a_cols, b_rows, b_cols, c_cols):
    # a_cols: list of 2 global col indices for A
    # b_rows: list of 2 global row indices for B
    # b_cols: list of 2 global col indices for B
    # c_cols: list of 2 global col indices for C
    
    # Local A indices: 00, 01, 10, 11
    # Global A idx: i*4 + j
    a_idx = [
        0*4 + a_cols[0], 0*4 + a_cols[1],
        1*4 + a_cols[0], 1*4 + a_cols[1]
    ]
    
    # Local B indices: 00, 01, 10, 11
    # Global B idx: j*4 + k
    b_idx = [
        b_rows[0]*4 + b_cols[0], b_rows[0]*4 + b_cols[1],
        b_rows[1]*4 + b_cols[0], b_rows[1]*4 + b_cols[1]
    ]
    
    # Local C indices: 00, 01, 10, 11
    # Global C idx: k*2 + i
    c_idx = [
        c_cols[0]*2 + 0, c_cols[1]*2 + 0,
        c_cols[0]*2 + 1, c_cols[1]*2 + 1
    ]
    
    # Strassen terms
    # 1. (a+d)(e+h)
    U[a_idx[0], col_start] += 1
    U[a_idx[3], col_start] += 1
    V[b_idx[0], col_start] += 1
    V[b_idx[3], col_start] += 1
    W[c_idx[0], col_start] += 1
    W[c_idx[3], col_start] += 1
    
    # 2. (c+d)e
    U[a_idx[2], col_start+1] += 1
    U[a_idx[3], col_start+1] += 1
    V[b_idx[0], col_start+1] += 1
    W[c_idx[2], col_start+1] += 1
    W[c_idx[3], col_start+1] -= 1
    
    # 3. a(f-h)
    U[a_idx[0], col_start+2] += 1
    V[b_idx[1], col_start+2] += 1
    V[b_idx[3], col_start+2] -= 1
    W[c_idx[1], col_start+2] += 1
    W[c_idx[3], col_start+2] += 1
    
    # 4. d(g-e)
    U[a_idx[3], col_start+3] += 1
    V[b_idx[2], col_start+3] += 1
    V[b_idx[0], col_start+3] -= 1
    W[c_idx[0], col_start+3] += 1
    W[c_idx[2], col_start+3] += 1
    
    # 5. (a+b)h
    U[a_idx[0], col_start+4] += 1
    U[a_idx[1], col_start+4] += 1
    V[b_idx[3], col_start+4] += 1
    W[c_idx[0], col_start+4] -= 1
    W[c_idx[1], col_start+4] += 1
    
    # 6. (c-a)(e+f)
    U[a_idx[2], col_start+5] += 1
    U[a_idx[0], col_start+5] -= 1
    V[b_idx[0], col_start+5] += 1
    V[b_idx[1], col_start+5] += 1
    W[c_idx[3], col_start+5] += 1
    
    # 7. (b-d)(g+h)
    U[a_idx[1], col_start+6] += 1
    U[a_idx[3], col_start+6] -= 1
    V[b_idx[2], col_start+6] += 1
    V[b_idx[3], col_start+6] += 1
    W[c_idx[0], col_start+6] += 1
    W[c_idx[3], col_start+6] += 1
