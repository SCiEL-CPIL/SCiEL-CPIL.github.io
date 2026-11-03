import numpy as np

def construct():
    n, m, p = 3, 5, 5
    total_rank = 36
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    # Helper to add a block
    def add_block(U_b, V_b, W_b, i_off, j_off, k_off, n_b, m_b, p_b):
        nonlocal col_offset
        rank_b = U_b.shape[1]
        
        # Map local rows to global rows
        # U_b rows: i_l * m_b + j_l
        for c in range(rank_b):
            for r in range(U_b.shape[0]):
                val = U_b[r, c]
                if val != 0:
                    i_l = r // m_b
                    j_l = r % m_b
                    i_g = i_l + i_off
                    j_g = j_l + j_off
                    U[i_g * m + j_g, col_offset + c] += val
                    
            for r in range(V_b.shape[0]):
                val = V_b[r, c]
                if val != 0:
                    j_l = r // p_b
                    k_l = r % p_b
                    j_g = j_l + j_off
                    k_g = k_l + k_off
                    V[j_g * p + k_g, col_offset + c] += val
                    
            for r in range(W_b.shape[0]):
                val = W_b[r, c]
                if val != 0:
                    k_l = r // n_b
                    i_l = r % n_b
                    k_g = k_l + k_off
                    i_g = i_l + i_off
                    W[k_g * n + i_g, col_offset + c] += val
                    
        col_offset += rank_b

    # Generate Strassen 2x2x2
    def strassen_2x2x2():
        n_b, m_b, p_b = 2, 2, 2
        r = 7
        U_b = np.zeros((n_b * m_b, r))
        V_b = np.zeros((m_b * p_b, r))
        W_b = np.zeros((p_b * n_b, r))
        
        # M1: (A00+A11)(B00+B11)
        U_b[0, 0] = 1; U_b[3, 0] = 1
        V_b[0, 0] = 1; V_b[3, 0] = 1
        W_b[0, 0] = 1; W_b[3, 0] = 1
        
        # M2: (A10+A11)B00
        U_b[2, 1] = 1; U_b[3, 1] = 1
        V_b[0, 1] = 1
        W_b[1, 1] = 1; W_b[3, 1] = -1
        
        # M3: A00(B10-B11)
        U_b[0, 2] = 1
        V_b[2, 2] = 1; V_b[3, 2] = -1
        W_b[2, 2] = 1; W_b[3, 2] = 1
        
        # M4: A11(B01-B00)
        U_b[3, 3] = 1
        V_b[1, 3] = 1; V_b[0, 3] = -1
        W_b[0, 3] = 1; W_b[1, 3] = 1
        
        # M5: (A00+A01)B11
        U_b[0, 4] = 1; U_b[1, 4] = 1
        V_b[3, 4] = 1
        W_b[0, 4] = -1; W_b[2, 4] = 1
        
        # M6: (A10-A00)(B00+B01)
        U_b[2, 5] = 1; U_b[0, 5] = -1
        V_b[0, 5] = 1; V_b[1, 5] = 1
        W_b[3, 5] = 1
        
        # M7: (A01-A11)(B10+B11)
        U_b[1, 6] = 1; U_b[3, 6] = -1
        V_b[2, 6] = 1; V_b[3, 6] = 1
        W_b[0, 6] = 1
        
        return U_b, V_b, W_b, n_b, m_b, p_b

    # Generate Rank-1 block
    def rank1_block(n_b, m_b, p_b):
        r = 1
        U_b = np.zeros((n_b * m_b, r))
        V_b = np.zeros((m_b * p_b, r))
        W_b = np.zeros((p_b * n_b, r))
        
        for i in range(n_b):
            for j in range(m_b):
                U_b[i * m_b + j, 0] = 1
        for j in range(m_b):
            for k in range(p_b):
                V_b[j * p_b + k, 0] = 1
        for k in range(p_b):
            for i in range(n_b):
                W_b[k * n_b + i, 0] = 1
                
        return U_b, V_b, W_b, n_b, m_b, p_b

    # Block 1: 2x2x2
    U1, V1, W1, n1, m1, p1 = strassen_2x2x2()
    add_block(U1, V1, W1, 0, 0, 0, n1, m1, p1)
    
    # Block 2: 2x2x3 (Split K into 1,2 -> M221 + M222)
    # M221
    U2a, V2a, W2a, n2a, m2a, p2a = rank1_block(2, 2, 1)
    add_block(U2a, V2a, W2a, 0, 0, 0, n2a, m2a, p2a)
    # M222
    U2b, V2b, W2b, n2b, m2b, p2b = strassen_2x2x2()
    add_block(U2b, V2b, W2b, 0, 0, 1, n2b, m2b, p2b)
    
    # Block 3: 2x3x2 (Split J into 1,2 -> M212 + M222)
    # M212
    U3a, V3a, W3a, n3a, m3a, p3a = rank1_block(2, 1, 2)
    add_block(U3a, V3a, W3a, 0, 0, 0, n3a, m3a, p3a)
    # M222
    U3b, V3b, W3b, n3b, m3b, p3b = strassen_2x2x2()
    add_block(U3b, V3b, W3b, 0, 1, 0, n3b, m3b, p3b)
    
    # Block 4: 2x3x3 (Split J into 1,2 -> M213 + M223)
    # M213
    U4a, V4a, W4a, n4a, m4a, p4a = rank1_block(2, 1, 3)
    add_block(U4a, V4a, W4a, 0, 0, 0, n4a, m4a, p4a)
    # M223 (M221 + M222)
    U4b1, V4b1, W4b1, n4b1, m4b1, p4b1 = rank1_block(2, 2, 1)
    add_block(U4b1, V4b1, W4b1, 0, 1, 0, n4b1, m4b1, p4b1)
    U4b2, V4b2, W4b2, n4b2, m4b2, p4b2 = strassen_2x2x2()
    add_block(U4b2, V4b2, W4b2, 0, 1, 1, n4b2, m4b2, p4b2)
    
    # Block 5: 1x2x2
    U5, V5, W5, n5, m5, p5 = rank1_block(1, 2, 2)
    add_block(U5, V5, W5, 2, 0, 0, n5, m5, p5)
    
    # Block 6: 1x2x3
    U6, V6, W6, n6, m6, p6 = rank1_block(1, 2, 3)
    add_block(U6, V6, W6, 2, 0, 0, n6, m6, p6)
    
    # Block 7: 1x3x2
    U7, V7, W7, n7, m7, p7 = rank1_block(1, 3, 2)
    add_block(U7, V7, W7, 2, 0, 0, n7, m7, p7)
    
    # Block 8: 1x3x3
    U8, V8, W8, n8, m8, p8 = rank1_block(1, 3, 3)
    add_block(U8, V8, W8, 2, 0, 0, n8, m8, p8)

    return {"U": U, "V": V, "W": W}
