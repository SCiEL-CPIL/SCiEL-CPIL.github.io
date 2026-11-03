def add_strassen_block(U, V, W, col_start, I, J, K):
    # I, J, K are lists of 2 global indices
    i0, i1 = I
    j0, j1 = J
    k0, k1 = K
    
    # Map local indices to global
    # Local 0 -> global 0, Local 1 -> global 1
    
    # Product 1: (A_i0_j0 + A_i1_j1)(B_j0_k0 + B_j1_k1)
    c = col_start
    U[i0*5 + j0][c] += 1
    U[i1*5 + j1][c] += 1
    V[j0*5 + k0][c] += 1
    V[j1*5 + k1][c] += 1
    W[k0*3 + i0][c] += 1
    W[k1*3 + i1][c] += 1
    c += 1
    
    # Product 2: (A_i1_j0 + A_i1_j1) B_j0_k0
    U[i1*5 + j0][c] += 1
    U[i1*5 + j1][c] += 1
    V[j0*5 + k0][c] += 1
    W[k0*3 + i1][c] += 1
    W[k1*3 + i1][c] += -1
    c += 1
    
    # Product 3: A_i0_j0 (B_j0_k1 - B_j1_k1)
    U[i0*5 + j0][c] += 1
    V[j0*5 + k1][c] += 1
    V[j1*5 + k1][c] += -1
    W[k1*3 + i0][c] += 1
    W[k1*3 + i1][c] += 1
    c += 1
    
    # Product 4: A_i1_j1 (B_j1_k0 - B_j0_k1)
    U[i1*5 + j1][c] += 1
    V[j1*5 + k0][c] += 1
    V[j0*5 + k1][c] += -1
    W[k0*3 + i0][c] += 1
    W[k0*3 + i1][c] += 1
    c += 1
    
    # Product 5: (A_i0_j0 + A_i0_j1) B_j1_k1
    U[i0*5 + j0][c] += 1
    U[i0*5 + j1][c] += 1
    V[j1*5 + k1][c] += 1
    W[k0*3 + i0][c] += -1
    W[k1*3 + i0][c] += 1
    c += 1
    
    # Product 6: (A_i1_j0 - A_i0_j0) (B_j0_k0 + B_j0_k1)
    U[i1*5 + j0][c] += 1
    U[i0*5 + j0][c] += -1
    V[j0*5 + k0][c] += 1
    V[j0*5 + k1][c] += 1
    W[k1*3 + i1][c] += 1
    c += 1
    
    # Product 7: (A_i0_j1 - A_i1_j1) (B_j1_k0 + B_j1_k1)
    U[i0*5 + j1][c] += 1
    U[i1*5 + j1][c] += -1
    V[j1*5 + k0][c] += 1
    V[j1*5 + k1][c] += 1
    W[k0*3 + i0][c] += 1
    c += 1
    
    return c
