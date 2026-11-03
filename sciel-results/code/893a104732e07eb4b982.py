import time

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Block splits
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    # Precompute offsets
    n_offsets = [0]
    for x in n_splits[:-1]:
        n_offsets.append(n_offsets[-1] + x)
    m_offsets = [0]
    for x in m_splits[:-1]:
        m_offsets.append(m_offsets[-1] + x)
    p_offsets = [0]
    for x in p_splits[:-1]:
        p_offsets.append(p_offsets[-1] + x)
        
    # Total rank calculation
    total_rank = 0
    block_ranks = []
    for i, ni in enumerate(n_splits):
        for j, mj in enumerate(m_splits):
            for k, pk in enumerate(p_splits):
                if ni == 2 and mj == 2 and pk == 2:
                    block_ranks.append(7)
                    total_rank += 7
                else:
                    r = ni * mj * pk
                    block_ranks.append(r)
                    total_rank += r
                    
    rank = total_rank
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen decomposition vectors for 2x2x2
    # A: [[a, b], [c, d]] -> [a, b, c, d]
    # B: [[e, f], [g, h]] -> [e, f, g, h]
    # C: [[C00, C01], [C10, C11]] -> [C00, C01, C10, C11]
    
    # M1 = (a+d)(e+h)
    U_M1 = [1, 0, 0, 1]
    V_M1 = [1, 0, 0, 1]
    W_M1 = [1, 1, 1, 1]
    
    # M2 = (c+d)e
    U_M2 = [0, 0, 1, 1]
    V_M2 = [1, 0, 0, 0]
    W_M2 = [0, 0, 1, 0] # C10 += M2
    
    # M3 = a(f-h)
    U_M3 = [1, 0, 0, 0]
    V_M3 = [0, 1, 0, -1]
    W_M3 = [0, 1, 0, 1] # C01 += M3, C11 += M3
    
    # M4 = d(g-e)
    U_M4 = [0, 0, 0, 1]
    V_M4 = [0, 0, 1, -1]
    W_M4 = [1, 0, 1, 0] # C00 += M4, C10 += M4
    
    # M5 = (a+b)h
    U_M5 = [1, 1, 0, 0]
    V_M5 = [0, 0, 0, 1]
    W_M5 = [-1, -1, 0, 0] # C00 -= M5, C01 -= M5
    
    # M6 = (c-a)(e+f)
    U_M6 = [-1, 0, 1, 0]
    V_M6 = [1, 1, 0, 0]
    W_M6 = [0, 0, 0, 1] # C11 += M6
    
    # M7 = (b-d)(g+h)
    U_M7 = [0, 1, 0, -1]
    V_M7 = [0, 0, 1, 1]
    W_M7 = [1, 0, 0, 1] # C00 += M7, C11 += M7
    
    strassen_terms = [
        (U_M1, V_M1, W_M1),
        (U_M2, V_M2, W_M2),
        (U_M3, V_M3, W_M3),
        (U_M4, V_M4, W_M4),
        (U_M5, V_M5, W_M5),
        (U_M6, V_M6, W_M6),
        (U_M7, V_M7, W_M7)
    ]
    
    for i, ni in enumerate(n_splits):
        for j, mj in enumerate(m_splits):
            for k, pk in enumerate(p_splits):
                
                I = n_offsets[i]
                J = m_offsets[j]
                K = p_offsets[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Add 7 Strassen terms
                    for U_loc, V_loc, W_loc in strassen_terms:
                        # U: 2x2 -> indices I+0..1, J+0..1
                        for r in range(2):
                            for c in range(2):
                                idx = r * 2 + c
                                val = U_loc[idx]
                                if val != 0:
                                    U[I + r][J + c + col] = val # Wait, U is (n*m, rank). Row is I+r? No.
                                    # U row index is i*m + j.
                                    # Global row for A is I+r. But U index is (I+r)*m + (J+c)?
                                    # No. U has shape (n*m, rank).
                                    # Row index in U is (global_i) * m + (global_j).
                                    # global_i = I + r. global_j = J + c.
                                    u_row = (I + r) * m + (J + c)
                                    U[u_row][col] = val
                                    
                        # V: 2x2 -> indices J+0..1, K+0..1
                        for r in range(2):
                            for c in range(2):
                                idx = r * 2 + c
                                val = V_loc[idx]
                                if val != 0:
                                    # V row index is (global_j) * p + (global_k).
                                    # global_j = J + r. global_k = K + c.
                                    v_row = (J + r) * p + (K + c)
                                    V[v_row][col] = val
                                    
                        # W: 2x2 -> indices K+0..1, I+0..1
                        for r in range(2):
                            for c in range(2):
                                idx = r * 2 + c
                                val = W_loc[idx]
                                if val != 0:
                                    # W row index is (global_k) * n + (global_i).
                                    # global_k = K + r. global_i = I + c.
                                    w_row = (K + r) * n + (I + c)
                                    W[w_row][col] = val
                                    
                        col += 1
                else:
                    # Schoolbook
                    # A: ni x mj
                    # B: mj x pk
                    # C: pk x ni
                    
                    for r in range(ni):
                        for c in range(mj):
                            for s in range(pk):
                                # Term: A[r, c] * B[c, s] -> C[s, r]
                                
                                # U
                                u_row = (I + r) * m + (J + c)
                                U[u_row][col] = 1
                                
                                # V
                                v_row = (J + c) * p + (K + s)
                                V[v_row][col] = 1
                                
                                # W
                                w_row = (K + s) * n + (I + r)
                                W[w_row][col] = 1
                                
                                col += 1
                                
    return {"U": U, "V": V, "W": W}
