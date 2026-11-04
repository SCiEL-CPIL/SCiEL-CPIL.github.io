def construct():
    import time
    start_time = time.time()
    DEADLINE = 100.0  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Fallback schoolbook
    def _schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Local 2x2x2 Strassen tensors
    U_strassen = np.zeros((4, 7))
    V_strassen = np.zeros((4, 7))
    W_strassen = np.zeros((4, 7))
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U_strassen[0, 0] = 1; U_strassen[3, 0] = 1
    V_strassen[0, 0] = 1; V_strassen[1, 0] = 1
    W_strassen[0, 0] = 1; W_strassen[3, 0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U_strassen[2, 1] = 1; U_strassen[3, 1] = 1
    V_strassen[0, 1] = 1
    W_strassen[1, 1] = 1; W_strassen[3, 1] = -1
    
    # r=2: M3 = A00(B10-B11)
    U_strassen[0, 2] = 1
    V_strassen[2, 2] = 1; V_strassen[3, 2] = -1
    W_strassen[2, 2] = 1; W_strassen[3, 2] = 1
    
    # r=3: M4 = A11(B01-B00)
    U_strassen[3, 3] = 1
    V_strassen[1, 3] = 1; V_strassen[0, 3] = -1
    W_strassen[0, 3] = 1; W_strassen[1, 3] = 1
    
    # r=4: M5 = (A00+A01)B11
    U_strassen[0, 4] = 1; U_strassen[1, 4] = 1
    V_strassen[3, 4] = 1
    W_strassen[0, 4] = -1; W_strassen[2, 4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B10)
    U_strassen[2, 5] = 1; U_strassen[0, 5] = -1
    V_strassen[0, 5] = 1; V_strassen[2, 5] = 1
    W_strassen[3, 5] = 1
    
    # r=6: M7 = (A01-A11)(B01+B11)
    U_strassen[1, 6] = 1; U_strassen[3, 6] = -1
    V_strassen[1, 6] = 1; V_strassen[3, 6] = 1
    W_strassen[0, 6] = 1
    
    # Verify Strassen locally
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = 0.0
                for r in range(7):
                    total += U_strassen[i*2+j, r] * V_strassen[j*2+k, r] * W_strassen[k*2+i, r]
                if abs(total - 1.0) > 1e-9:
                    return _schoolbook(n, m, p)
                
    # Local 2x2x1 Schoolbook tensor (rank 4)
    U_sb = np.zeros((4, 4))
    V_sb = np.zeros((4, 4))
    W_sb = np.zeros((2, 4))
    
    # r=0: A00 B00 -> C0
    U_sb[0, 0] = 1; V_sb[0, 0] = 1; W_sb[0, 0] = 1
    # r=1: A01 B10 -> C0
    U_sb[1, 1] = 1; V_sb[2, 1] = 1; W_sb[0, 1] = 1
    # r=2: A10 B00 -> C1
    U_sb[2, 2] = 1; V_sb[0, 2] = 1; W_sb[1, 2] = 1
    # r=3: A11 B10 -> C1
    U_sb[3, 3] = 1; V_sb[2, 3] = 1; W_sb[1, 3] = 1
    
    # Verify Schoolbook locally
    for i in range(2):
        for j in range(2):
            for k in range(1):
                total = 0.0
                for r in range(4):
                    total += U_sb[i*2+j, r] * V_sb[j*2+k, r] * W_sb[k*2+i, r]
                if abs(total - 1.0) > 1e-9:
                    return _schoolbook(n, m, p)

    # Global Assembly
    m_chunks = [2, 2]
    p_chunks = [2, 2, 1]
    
    total_rank = 0
    for mc in m_chunks:
        for pc in p_chunks:
            if pc == 2:
                total_rank += 7
            else:
                total_rank += 4
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    offset = 0
    for m_idx, m_size in enumerate(m_chunks):
        for p_idx, p_size in enumerate(p_chunks):
            if p_size == 2:
                rank_block = 7
                U_loc = U_strassen
                V_loc = V_strassen
                W_loc = W_strassen
            else:
                rank_block = 4
                U_loc = U_sb
                V_loc = V_sb
                W_loc = W_sb
            
            for r in range(rank_block):
                for i in range(2):
                    for j in range(2):
                        g_u_row = i*m + (m_idx*m_size + j)
                        l_u_row = i*2 + j
                        U[g_u_row, offset + r] += U_loc[l_u_row, r]
                
                for j in range(2):
                    for k in range(p_size):
                        g_v_row = (m_idx*m_size + j)*p + (p_idx*p_size + k)
                        l_v_row = j*2 + k
                        V[g_v_row, offset + r] += V_loc[l_v_row, r]
                
                for i in range(2):
                    for k in range(p_size):
                        g_w_row = (p_idx*p_size + k)*n + i
                        l_w_row = k*2 + i
                        W[g_w_row, offset + r] += W_loc[l_w_row, r]
            
            offset += rank_block
            
            if time.time() > DEADLINE:
                return _schoolbook(n, m, p)

    # Final Verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(total_rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(total - 1.0) > 1e-9:
                    return _schoolbook(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
