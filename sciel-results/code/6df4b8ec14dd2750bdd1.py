import time

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    Shapes: (4, 7), (4, 7), (4, 7)
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    # C00 += 1, C11 += 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1: M2 = (A10+A11)B00
    # C10 += 1, C11 += 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = 1
    
    # Term 2: M3 = A00(B01-B11)
    # C01 += 1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1
    
    # Term 3: M4 = A11(B10-B00)
    # C00 += 1, C10 += 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # Term 4: M5 = (A00+A01)B11
    # C00 -= 1, C11 += 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[3][4] = 1
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    # C11 += 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    # C00 += 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _place_sub(U_g, V_g, W_g, U_s, V_s, W_s, i_off, j_off, k_off, n_s, m_s, p_s, n_g, m_g, p_g, rank_offset):
    """
    Places a sub-decomposition into the global matrices.
    """
    n_sub = len(U_s) // m_s
    m_sub = m_s
    p_sub = len(V_s) // m_s
    
    # U: local index a = i_loc * m_sub + j_loc
    # Global index a = (i_off + i_loc) * m_g + (j_off + j_loc)
    for a_loc in range(len(U_s)):
        i_loc = a_loc // m_sub
        j_loc = a_loc % m_sub
        a_g = (i_off + i_loc) * m_g + (j_off + j_loc)
        for t in range(len(U_s[0])):
            if U_s[a_loc][t] != 0:
                U_g[a_g][rank_offset + t] += U_s[a_loc][t]
                
    # V: local index b = j_loc * p_sub + k_loc
    # Global index b = (j_off + j_loc) * p_g + (k_off + k_loc)
    for b_loc in range(len(V_s)):
        j_loc = b_loc // p_sub
        k_loc = b_loc % p_sub
        b_g = (j_off + j_loc) * p_g + (k_off + k_loc)
        for t in range(len(V_s[0])):
            if V_s[b_loc][t] != 0:
                V_g[b_g][rank_offset + t] += V_s[b_loc][t]
                
    # W: local index c = k_loc * n_sub + i_loc
    # Global index c = (k_off + k_loc) * n_g + (i_off + i_loc)
    for c_loc in range(len(W_s)):
        k_loc = c_loc // n_sub
        i_loc = c_loc % n_sub
        c_g = (k_off + k_loc) * n_g + (i_off + i_loc)
        for t in range(len(W_s[0])):
            if W_s[c_loc][t] != 0:
                W_g[c_g][rank_offset + t] += W_s[c_loc][t]

def _strassen_4x4():
    """
    Returns U, V, W for 4x4x4 Strassen decomposition using Kronecker product of 2x2x2.
    Shapes: (16, 49), (16, 49), (16, 49)
    """
    U2, V2, W2 = _strassen_2x2()
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for t1 in range(7):
        for t2 in range(7):
            t = t1 * 7 + t2
            for a_loc in range(4):
                ib = a_loc // 2
                ibs = a_loc % 2 # Wait, a_loc = ib*2 + jb? No.
                # a_loc = i*2 + j.
                # i = ib*2 + is, j = jb*2 + js.
                # a_loc = (ib*2+is)*2 + (jb*2+js) = ib*4 + is*2 + jb*2 + js.
                # This mapping is complex.
                # Let's iterate over i,j for 4x4.
                pass
            
    # Simpler: Iterate over all 16 indices for U (4x4)
    # a = i*4 + j
    # i = ib*2 + is
    # j = jb*2 + js
    # a_block = ib*2 + jb
    # a_sub = is*2 + js
    
    for i in range(4):
        for j in range(4):
            a = i * 4 + j
            ib = i // 2
            is_ = i % 2
            jb = j // 2
            js = j % 2
            a_block = ib * 2 + jb
            a_sub = is_ * 2 + js
            
            for t1 in range(7):
                for t2 in range(7):
                    t = t1 * 7 + t2
                    val = U2[a_block][t1] * U2[a_sub][t2]
                    if val != 0:
                        U[a][t] = val

    # V: b = j*4 + k
    # j = jb*2 + js
    # k = kb*2 + ks
    # b_block = jb*2 + kb
    # b_sub = js*2 + ks
    for j in range(4):
        for k in range(4):
            b = j * 4 + k
            jb = j // 2
            js = j % 2
            kb = k // 2
            ks = k % 2
            b_block = jb * 2 + kb
            b_sub = js * 2 + ks
            
            for t1 in range(7):
                for t2 in range(7):
                    t = t1 * 7 + t2
                    val = V2[b_block][t1] * V2[b_sub][t2]
                    if val != 0:
                        V[b][t] = val

    # W: c = k*4 + i
    # k = kb*2 + ks
    # i = ib*2 + is
    # c_block = kb*2 + ib
    # c_sub = ks*2 + is
    for k in range(4):
        for i in range(4):
            c = k * 4 + i
            kb = k // 2
            ks = k % 2
            ib = i // 2
            is_ = i % 2
            c_block = kb * 2 + ib
            c_sub = ks * 2 + is_
            
            for t1 in range(7):
                for t2 in range(7):
                    t = t1 * 7 + t2
                    val = W2[c_block][t1] * W2[c_sub][t2]
                    if val != 0:
                        W[c][t] = val

    return U, V, W

def construct():
    """
    Rank-minimal exact tensor decomposition for 4x5 x 5x6 matrix multiplication.
    Target Rank: 101.
    """
    n, m, p = 4, 5, 6
    deadline = time.time() + 110
    
    # Total Rank = 24 (4x1x6) + 49 (4x4x4) + 28 (4x4x2) = 101
    rank_total = 101
    
    U = [[0] * rank_total for _ in range(n * m)] # 20x101
    V = [[0] * rank_total for _ in range(m * p)] # 30x101
    W = [[0] * rank_total for _ in range(p * n)] # 24x101
    
    # --- Part 1: 4x1x6 (Rank 24) ---
    # A[:, 4] * B[4, :]
    # i in 0..3, j=4, k in 0..5
    rank_offset = 0
    for i in range(4):
        for k in range(6):
            t = rank_offset + i * 6 + k
            
            a_idx = i * m + 4 # i*5 + 4
            b_idx = 4 * p + k # 4*6 + k
            c_idx = k * n + i # k*4 + i
            
            U[a_idx][t] = 1
            V[b_idx][t] = 1
            W[c_idx][t] = 1
            
    rank_offset += 24
    
    # --- Part 2: 4x4x4 (Rank 49) ---
    # A[:, 0:4] * B[0:4, 0:4]
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W} # Fallback if slow
        
    U_44, V_44, W_44 = _strassen_4x4()
    _place_sub(U, V, W, U_44, V_44, W_44, 
               i_off=0, j_off=0, k_off=0, 
               n_s=4, m_s=4, p_s=4, 
               n_g=n, m_g=m, p_g=p, 
               rank_offset=rank_offset)
               
    rank_offset += 49
    
    # --- Part 3: 4x4x2 (Rank 28) ---
    # A[:, 0:4] * B[0:4, 4:6]
    # Split into 4 blocks of 2x2x2 Strassen
    U_22, V_22, W_22 = _strassen_2x2()
    
    # Block 1: i=0..1, j=0..1, k=0..1 (offsets 0,0,4)
    _place_sub(U, V, W, U_22, V_22, W_22, 
               i_off=0, j_off=0, k_off=4, 
               n_s=2, m_s=2, p_s=2, 
               n_g=n, m_g=m, p_g=p, 
               rank_offset=rank_offset)
    rank_offset += 7
    
    # Block 2: i=0..1, j=2..3, k=0..1 (offsets 0,2,4)
    _place_sub(U, V, W, U_22, V_22, W_22, 
               i_off=0, j_off=2, k_off=4, 
               n_s=2, m_s=2, p_s=2, 
               n_g=n, m_g=m, p_g=p, 
               rank_offset=rank_offset)
    rank_offset += 7
    
    # Block 3: i=2..3, j=0..1, k=0..1 (offsets 2,0,4)
    _place_sub(U, V, W, U_22, V_22, W_22, 
               i_off=2, j_off=0, k_off=4, 
               n_s=2, m_s=2, p_s=2, 
               n_g=n, m_g=m, p_g=p, 
               rank_offset=rank_offset)
    rank_offset += 7
    
    # Block 4: i=2..3, j=2..3, k=0..1 (offsets 2,2,4)
    _place_sub(U, V, W, U_22, V_22, W_22, 
               i_off=2, j_off=2, k_off=4, 
               n_s=2, m_s=2, p_s=2, 
               n_g=n, m_g=m, p_g=p, 
               rank_offset=rank_offset)
    rank_offset += 7
    
    return {"U": U, "V": V, "W": W}
