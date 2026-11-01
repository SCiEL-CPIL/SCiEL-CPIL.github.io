def add_strassen_block(U, V, W, a_i, a_j, b_j, b_k, w_k, w_i, col_offset):
    # a_i, a_j are lists of indices for A
    # b_j, b_k are lists of indices for B
    # w_k, w_i are lists of indices for W (C)
    # We assume 2x2 blocks
    # a_i[0], a_i[1] are rows of A
    # a_j[0], a_j[1] are cols of A
    # b_j[0], b_j[1] are rows of B (same as cols of A)
    # b_k[0], b_k[1] are cols of B
    # w_k[0], w_k[1] are cols of C
    # w_i[0], w_i[1] are rows of C
    
    # M1
    col = col_offset
    # U
    U[a_i[1]*4 + a_j[1]][col] += 1
    U[a_i[0]*4 + a_j[0]][col] -= 1
    # V
    V[b_j[0]*5 + b_k[0]][col] += 1
    V[b_j[1]*5 + b_k[1]][col] += 1
    # W
    W[w_k[0]*2 + w_i[0]][col] += 1
    W[w_k[1]*2 + w_i[1]][col] += 1
