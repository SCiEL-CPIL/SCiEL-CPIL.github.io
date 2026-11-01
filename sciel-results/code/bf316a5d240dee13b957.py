def construct():
    # Rank 49 decomposition using Kronecker product of rank-7 Strassen decomposition
    # T4 = T2 (x) T2
    
    # Define the 7 vectors for MM_2 (rank 7)
    # u_s, v_s, w_s are length 4 vectors corresponding to 2x2 matrices
    # Indexing: 0:00, 1:01, 2:10, 3:11
    
    u_vecs = [
        [1, 0, 0, 1],  # P1: a00 + a11
        [0, 0, 1, 1],  # P2: a10 + a11
        [1, 0, 0, 0],  # P3: a00
        [0, 0, 0, 1],  # P4: a11
        [1, 1, 0, 0],  # P5: a00 + a01
        [-1, 0, 1, 0], # P6: -a00 + a10
        [0, 1, 0, -1]  # P7: a01 - a11
    ]
    v_vecs = [
        [1, 0, 0, 1],  # P1: b00 + b11
        [1, 0, 0, 0],  # P2: b00
        [0, 1, 0, -1], # P3: b01 - b11
        [0, -1, 1, 0], # P4: -b00 + b10
        [0, 0, 0, 1],  # P5: b11
        [1, 1, 0, 0],  # P6: b00 + b01
        [0, 0, 1, 1]   # P7: b10 + b11
    ]
    w_vecs = [
        [1, 0, 0, 1],  # P1: c00 + c11
        [0, 0, 1, 0],  # P2: c10
        [0, 1, 0, -1], # P3: c01 - c11
        [1, 0, 1, 0],  # P4: c00 + c10
        [0, 1, 0, 1],  # P5: c01 + c11
        [-1, 0, 0, 0], # P6: -c00
        [1, 0, 0, -1]  # P7: c00 - c11
    ]
    
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # Fill U, V, W
    # Index mapping:
    # i in 0..3, j in 0..3
    # i = 2*i1 + i2, j = 2*j1 + j2
    # U[4*i + j][7*s + t] = u_vecs[s][2*i1 + j1] * u_vecs[t][2*i2 + j2]
    # Wait, u_vecs[s] is length 4. Index is 2*r + c.
    # For A[i, j], the "block" indices are i1, j1. The "inner" indices are i2, j2.
    # The Kronecker product u_s (x) u_t has component at (i1, j1, i2, j2) equal to u_s[i1, j1] * u_t[i2, j2].
    # The flattened index for A is 4*i + j.
    # i = 2*i1 + i2, j = 2*j1 + j2.
    # So U[4*(2*i1+i2) + (2*j1+j2)][7*s+t] = u_vecs[s][2*i1+j1] * u_vecs[t][2*i2+j2]
    
    for i in range(4):
        for j in range(4):
            i1 = i // 2
            i2 = i % 2
            j1 = j // 2
            j2 = j % 2
            idx_u = 4 * i + j
            
            for s in range(7):
                for t in range(7):
                    rank_idx = 7 * s + t
                    # u_vecs[s] index for block (i1, j1) is 2*i1 + j1
                    # u_vecs[t] index for block (i2, j2) is 2*i2 + j2
                    U[idx_u][rank_idx] = u_vecs[s][2 * i1 + j1] * u_vecs[t][2 * i2 + j2]
                    
                    # V[j, k]
                    # j = 2*j1 + j2, k = 2*k1 + k2
                    # V[4*j + k][7*s + t] = v_vecs[s][2*j1 + k1] * v_vecs[t][2*j2 + k2]
                    # We need to loop over k as well? No, V is 16x49.
                    # The row index is 4*j + k.
                    # So for each j, k, we compute.
                    
            # It is more efficient to loop over s, t first or fill in nested loops properly.
            # Let's restart the filling logic to be clean.
            
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for s in range(7):
        for t in range(7):
            rank_idx = 7 * s + t
            for i in range(4):
                i1 = i // 2
                i2 = i % 2
                for j in range(4):
                    j1 = j // 2
                    j2 = j % 2
                    idx_u = 4 * i + j
                    U[idx_u][rank_idx] = u_vecs[s][2 * i1 + j1] * u_vecs[t][2 * i2 + j2]
                    
                    # V row index is 4*j + k. k is the other dimension.
                    # We need to fill V[4*j + k] for all k.
                    # So we should loop k here or in a separate loop.
                    # Let's do V and W in separate loops for clarity.
                    
    # Fill V
    for s in range(7):
        for t in range(7):
            rank_idx = 7 * s + t
            for j in range(4):
                j1 = j // 2
                j2 = j % 2
                for k in range(4):
                    k1 = k // 2
                    k2 = k % 2
                    idx_v = 4 * j + k
                    V[idx_v][rank_idx] = v_vecs[s][2 * j1 + k1] * v_vecs[t][2 * j2 + k2]
                    
    # Fill W
    # W row index is 4*k + i (Trace convention W[k*n+i])
    for s in range(7):
        for t in range(7):
            rank_idx = 7 * s + t
            for k in range(4):
                k1 = k // 2
                k2 = k % 2
                for i in range(4):
                    i1 = i // 2
                    i2 = i % 2
                    idx_w = 4 * k + i
                    W[idx_w][rank_idx] = w_vecs[s][2 * k1 + i1] * w_vecs[t][2 * k2 + i2]
                    
    return {"U": U, "V": V, "W": W}
