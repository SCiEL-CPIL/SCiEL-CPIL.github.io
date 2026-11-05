import time

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # Safety margin
    
    # 1. Strassen 4x4x4
    U_str, V_str, W_str = strassen_factors(4)
    
    # 2. Initialize full factors
    n, m, p = 4, 5, 5
    rank_str = 49
    rank_b = 16
    rank_c = 20
    total_rank = rank_str + rank_b + rank_c
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # 3. Fill Strassen part (cols 0-48)
    # Strassen U, V, W are for 4x4, 4x4, 4x4.
    # U_str indices: i*4 + j. Global U indices: i*5 + j (j in 0..3).
    # V_str indices: j*4 + k. Global V indices: j*5 + k (k in 0..3).
    # W_str indices: k*4 + i. Global W indices: k*4 + i (k in 0..3, i in 0..3).
    
    for i in range(4):
        for j in range(4):
            u_idx_global = i * 5 + j
            u_idx_local = i * 4 + j
            for r in range(rank_str):
                U[u_idx_global][r] = U_str[u_idx_local][r]
                
    for j in range(4):
        for k in range(4):
            v_idx_global = j * 5 + k
            v_idx_local = j * 4 + k
            for r in range(rank_str):
                V[v_idx_global][r] = V_str[v_idx_local][r]
                
    for k in range(4):
        for i in range(4):
            w_idx_global = k * 4 + i
            w_idx_local = k * 4 + i
            for r in range(rank_str):
                W[w_idx_global][r] = W_str[w_idx_local][r]
                
    # 4. Fill Part B (cols 49-64)
    # A1 (4x4) * B12 (4x1) -> C12 (4x1)
    # j in 0..3, k=4
    col_offset_b = rank_str
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            u_idx = i * 5 + j
            v_idx = j * 5 + 4
            w_idx = 4 * 4 + i
            U[u_idx][col_offset_b + idx] = 1
            V[v_idx][col_offset_b + idx] = 1
            W[w_idx][col_offset_b + idx] = 1
            
    # 5. Fill Part C (cols 65-84)
    # A2 (4x1) * B2 (1x5) -> C2 (4x5)
    # j=4, k in 0..4
    col_offset_c = rank_str + rank_b
    for i in range(4):
        for k in range(5):
            idx = i * 5 + k
            u_idx = i * 5 + 4
            v_idx = 4 * 5 + k
            w_idx = k * 4 + i
            U[u_idx][col_offset_c + idx] = 1
            V[v_idx][col_offset_c + idx] = 1
            W[w_idx][col_offset_c + idx] = 1
            
    if time.time() > deadline:
        # Should not happen, but just in case
        pass
        
    return {"U": U, "V": V, "W": W}
