def _block_decompose_4x5x5():
    n, m, p = 4, 5, 5
    n1, n2 = 2, 2
    p1, p2 = 2, 3
    
    # 1. 2x5 * 5x2 (Rank 19)
    U1, V1, W1 = _decompose_2x5x2()
    # 2. 2x5 * 5x3 (Rank 30)
    U2, V2, W2 = _schoolbook(2, 5, 3)
    # 3. 2x5 * 5x2 (Rank 19)
    U3, V3, W3 = _decompose_2x5x2()
    # 4. 2x5 * 5x3 (Rank 30)
    U4, V4, W4 = _schoolbook(2, 5, 3)
    
    total_rank = 19 + 30 + 19 + 30
    U_full = np.zeros((n * m, total_rank), dtype=float)
    V_full = np.zeros((m * p, total_rank), dtype=float)
    W_full = np.zeros((p * n, total_rank), dtype=float)
    
    col_offset = 0
    
    # 1. Top-Left: A[0:2, :] * B[:, 0:2]
    # U1 shape (10, 19), V1 shape (10, 19), W1 shape (4, 19)
    # U_full[i*5+j, col] = U1[i*5+j, c] for i in 0..1, j in 0..4
    # V_full[j*5+k, col] = V1[j*2+k, c] for j in 0..4, k in 0..1
    # W_full[k*4+i, col] = W1[k*2+i, c] for k in 0..1, i in 0..1
    
    for c in range(19):
        for i in range(2):
            for j in range(5):
                U_full[i * 5 + j, col_offset + c] = U1[i * 5 + j, c]
        for j in range(5):
            for k in range(2):
                V_full[j * 5 + k, col_offset + c] = V1[j * 2 + k, c]
        for k in range(2):
            for i in range(2):
                W_full[k * 4 + i, col_offset + c] = W1[k * 2 + i, c]
    col_offset += 19
    
    # 2. Top-Right: A[0:2, :] * B[:, 2:5]
    # U2 shape (10, 30), V2 shape (15, 30), W2 shape (6, 30)
    # U_full[i*5+j, col] = U2[i*5+j, c] for i in 0..1, j in 0..4
    # V_full[j*5+k, col] = V2[j*3+k, c] for j in 0..4, k in 0..2 (local k)
    #   Global k = 2 + local_k
    # W_full[k*4+i, col] = W2[k*2+i, c] for k in 0..2 (local), i in 0..1
    #   Global k = 2 + local_k
    
    for c in range(30):
        for i in range(2):
            for j in range(5):
                U_full[i * 5 + j, col_offset + c] = U2[i * 5 + j, c]
        for j in range(5):
            for k_local in range(3):
                k_global = 2 + k_local
                V_full[j * 5 + k_global, col_offset + c] = V2[j * 3 + k_local, c]
        for k_local in range(3):
            k_global = 2 + k_local
            for i in range(2):
                W_full[k_global * 4 + i, col_offset + c] = W2[k_local * 2 + i, c]
    col_offset += 30
    
    # 3. Bottom-Left: A[2:4, :] * B[:, 0:2]
    # U3 shape (10, 19), V3 shape (10, 19), W3 shape (4, 19)
    # U_full[i*5+j, col] = U3[i*5+j, c] for i in 0..1 (local), j in 0..4
    #   Global i = 2 + local_i
    # V_full[j*5+k, col] = V3[j*2+k, c] for j in 0..4, k in 0..1
    # W_full[k*4+i, col] = W3[k*2+i, c] for k in 0..1, i in 0..1 (local)
    #   Global i = 2 + local_i
    
    for c in range(19):
        for i_local in range(2):
            i_global = 2 + i_local
            for j in range(5):
                U_full[i_global * 5 + j, col_offset + c] = U3[i_local * 5 + j, c]
        for j in range(5):
            for k in range(2):
                V_full[j * 5 + k, col_offset + c] = V3[j * 2 + k, c]
        for k in range(2):
            for i_local in range(2):
                i_global = 2 + i_local
                W_full[k * 4 + i_global, col_offset + c] = W3[k * 2 + i_local, c]
    col_offset += 19
    
    # 4. Bottom-Right: A[2:4, :] * B[:, 2:5]
    # U4 shape (10, 30), V4 shape (15, 30), W4 shape (6, 30)
    # U_full[i*5+j, col] = U4[i*5+j, c] for i in 0..1 (local), j in 0..4
    #   Global i = 2 + local_i
    # V_full[j*5+k, col] = V4[j*3+k, c] for j in 0..4, k in 0..2 (local)
    #   Global k = 2 + local_k
    # W_full[k*4+i, col] = W4[k*2+i, c] for k in 0..2 (local), i in 0..1 (local)
    #   Global k = 2 + local_k, Global i = 2 + local_i
    
    for c in range(30):
        for i_local in range(2):
            i_global = 2 + i_local
            for j in range(5):
                U_full[i_global * 5 + j, col_offset + c] = U4[i_local * 5 + j, c]
        for j in range(5):
            for k_local in range(3):
                k_global = 2 + k_local
                V_full[j * 5 + k_global, col_offset + c] = V4[j * 3 + k_local, c]
        for k_local in range(3):
            k_global = 2 + k_local
            for i_local in range(2):
                i_global = 2 + i_local
                W_full[k_global * 4 + i_global, col_offset + c] = W4[k_local * 2 + i_local, c]
    col_offset += 30
    
    assert col_offset == total_rank
    return U_full, V_full, W_full
