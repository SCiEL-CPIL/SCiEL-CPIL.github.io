def construct():
    n, m, p = DIMENSIONS # 4, 4, 8
    
    U4, V4, W4 = _build_4x4x4_decomposition()
    rank_4x4 = 49
    
    # U is 16 x 98
    # V is 32 x 98
    # W is 32 x 98
    
    U = [[0] * (2 * rank_4x4) for _ in range(16)]
    V = [[0] * (2 * rank_4x4) for _ in range(32)]
    W = [[0] * (2 * rank_4x4) for _ in range(32)]
    
    for r in range(rank_4x4):
        # Left half (k=0..3)
        # U is same
        for i in range(16):
            U[i][r] = U4[i][r]
            
        # V left: j*p + k. p=8. k=0..3.
        # V4 index: j*4 + k_local. k_local=0..3.
        # V_full index: j*8 + k_local.
        for j in range(4):
            for k_local in range(4):
                v_idx = j * 8 + k_local
                v4_idx = j * 4 + k_local
                V[v_idx][r] = V4[v4_idx][r]
                
        # W left: k*n + i. k=0..3.
        # W4 index: k_local*4 + i.
        # W_full index: k_local*4 + i.
        for k_local in range(4):
            for i in range(4):
                w_idx = k_local * 4 + i
                w4_idx = k_local * 4 + i
                W[w_idx][r] = W4[w4_idx][r]
                
        # Right half (k=4..7)
        r_right = r + rank_4x4
        
        for i in range(16):
            U[i][r_right] = U4[i][r]
            
        # V right: j*p + k. k=4..7.
        # k = 4 + k_local.
        for j in range(4):
            for k_local in range(4):
                k = 4 + k_local
                v_idx = j * 8 + k
                v4_idx = j * 4 + k_local
                V[v_idx][r_right] = V4[v4_idx][r]
                
        # W right: k*n + i. k=4..7.
        # k = 4 + k_local.
        for k_local in range(4):
            k = 4 + k_local
            for i in range(4):
                w_idx = k * 4 + i
                w4_idx = k_local * 4 + i
                W[w_idx][r_right] = W4[w4_idx][r]
                
    return {"U": U, "V": V, "W": W}
