import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Helper to create a block decomposition
    # This function returns U, V, W for a specific block of dimensions n_block, m_block, p_block
    # embedded in the full space.
    
    # We will build the full U, V, W directly.
    
    U_full = np.zeros((n*m, 0))
    V_full = np.zeros((m*p, 0))
    W_full = np.zeros((p*n, 0))
    
    def add_block(U_block, V_block, W_block, i_offset, j_offset, k_offset, n_b, m_b, p_b):
        # U_block shape: (n_b*m_b, rank)
        # V_block shape: (m_b*p_b, rank)
        # W_block shape: (p_b*n_b, rank)
        
        # We need to place these into U_full, V_full, W_full
        # U_full indices: i*m + j. i in [i_offset, i_offset+n_b), j in [j_offset, j_offset+m_b)
        # V_full indices: j*p + k. j in [j_offset, j_offset+m_b), k in [k_offset, k_offset+p_b)
        # W_full indices: k*n + i. k in [k_offset, k_offset+p_b), i in [i_offset, i_offset+n_b)
        
        rank = U_block.shape[1]
        
        U_new = np.zeros((n*m, rank))
        V_new = np.zeros((m*p, rank))
        W_new = np.zeros((p*n, rank))
        
        for i in range(n_b):
            for j in range(m_b):
                u_idx = (i_offset + i) * m + (j_offset + j)
                u_b_idx = i * m_b + j
                U_new[u_idx, :] = U_block[u_b_idx, :]
                
        for j in range(m_b):
            for k in range(p_b):
                v_idx = (j_offset + j) * p + (k_offset + k)
                v_b_idx = j * p_b + k
                V_new[v_idx, :] = V_block[v_b_idx, :]
                
        for k in range(p_b):
            for i in range(n_b):
                w_idx = (k_offset + k) * n + (i_offset + i)
                w_b_idx = k * n_b + i
                W_new[w_idx, :] = W_block[w_b_idx, :]
                
        return np.hstack([U_full, U_new]), np.hstack([V_full, V_new]), np.hstack([W_full, W_new])

    # Strassen 2x2 decomposition
    def strassen_2x2():
        # Returns U, V, W for M_{2,2,2}
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3 = A11(B12-B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4 = A22(B21-B11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1
        
        # M5 = (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[3, 4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[3, 6] = 1
        
        return U, V, W

    # Schoolbook 2x1x2 (Rank 4)
    def schoolbook_212():
        U = np.zeros((2, 4))
        V = np.zeros((1, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[0, 1] = 1; V[0, 1] = 1; W[1, 1] = 1
        U[1, 2] = 1; V[0, 2] = 1; W[0, 2] = 1
        U[1, 3] = 1; V[0, 3] = 1; W[1, 3] = 1
        return U, V, W

    # Schoolbook 2x2x1 (Rank 4)
    def schoolbook_221():
        U = np.zeros((4, 4))
        V = np.zeros((4, 4))
        W = np.zeros((1, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[2, 2] = 1; W[0, 2] = 1
        U[3, 3] = 1; V[3, 3] = 1; W[0, 3] = 1
        return U, V, W

    # Schoolbook 2x1x1 (Rank 2)
    def schoolbook_211():
        U = np.zeros((2, 2))
        V = np.zeros((1, 2))
        W = np.zeros((1, 2))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[0, 1] = 1; W[0, 1] = 1
        return U, V, W

    # Schoolbook 1x2x2 (Rank 4)
    def schoolbook_122():
        U = np.zeros((1, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
        U[0, 2] = 1; V[2, 2] = 1; W[0, 2] = 1
        U[0, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        return U, V, W

    # Schoolbook 1x1x2 (Rank 2)
    def schoolbook_112():
        U = np.zeros((1, 2))
        V = np.zeros((1, 2))
        W = np.zeros((2, 2))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[0, 1] = 1; V[0, 1] = 1; W[1, 1] = 1
        return U, U, W # Note: U and V are same shape and values in this case? No, U is 1x2, V is 1x2.

    # Schoolbook 1x2x1 (Rank 2)
    def schoolbook_121():
        U = np.zeros((1, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[0, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        return U, V, W

    # Schoolbook 1x1x1 (Rank 1)
    def schoolbook_111():
        U = np.ones((1, 1))
        V = np.ones((1, 1))
        W = np.ones((1, 1))
        return U, V, W

    # Construct M_{3,3,3} using blocks
    def get_333_decomp():
        U_333 = np.zeros((9, 0))
        V_333 = np.zeros((9, 0))
        W_333 = np.zeros((9, 0))
        
        # Block 11: 2x2x2 (Strassen)
        U11, V11, W11 = strassen_2x2()
        U_333, V_333, W_333 = add_block(U11, V11, W11, 0, 0, 0, 2, 2, 2)
        
        # Block 12: 2x1x2 (Schoolbook)
        U12, V12, W12 = schoolbook_212()
        U_333, V_333, W_333 = add_block(U12, V12, W12, 0, 2, 0, 2, 1, 2)
        
        # Block 21: 1x2x2 (Schoolbook)
        U21, V21, W21 = schoolbook_122()
        U_333, V_333, W_333 = add_block(U21, V21, W21, 2, 0, 0, 1, 2, 2)
        
        # Block 22: 1x1x2 (Schoolbook)
        U22, V22, W22 = schoolbook_112()
        U_333, V_333, W_333 = add_block(U22, V22, W22, 2, 2, 0, 1, 1, 2)
        
        # Block 13: 2x2x1 (Schoolbook)
        U13, V13, W13 = schoolbook_221()
        U_333, V_333, W_333 = add_block(U13, V13, W13, 0, 0, 2, 2, 2, 1)
        
        # Block 14: 2x1x1 (Schoolbook)
        U14, V14, W14 = schoolbook_211()
        U_333, V_333, W_333 = add_block(U14, V14, W14, 0, 2, 2, 2, 1, 1)
        
        # Block 23: 1x2x1 (Schoolbook)
        U23, V23, W23 = schoolbook_121()
        U_333, V_333, W_333 = add_block(U23, V23, W23, 2, 0, 2, 1, 2, 1)
        
        # Block 24: 1x1x1 (Schoolbook)
        U24, V24, W24 = schoolbook_111()
        U_333, V_333, W_333 = add_block(U24, V24, W24, 2, 2, 2, 1, 1, 1)
        
        return U_333, V_333, W_333

    # Construct M_{3,3,2} using blocks
    def get_332_decomp():
        U_332 = np.zeros((9, 0))
        V_332 = np.zeros((6, 0))
        W_332 = np.zeros((6, 0))
        
        # Block 11: 2x2x2 (Strassen)
        U11, V11, W11 = strassen_2x2()
        U_332, V_332, W_332 = add_block(U11, V11, W11, 0, 0, 0, 2, 2, 2)
        
        # Block 12: 2x1x2 (Schoolbook)
        U12, V12, W12 = schoolbook_212()
        U_332, V_332, W_332 = add_block(U12, V12, W12, 0, 2, 0, 2, 1, 2)
        
        # Block 21: 1x2x2 (Schoolbook)
        U21, V21, W21 = schoolbook_122()
        U_332, V_332, W_332 = add_block(U21, V21, W21, 2, 0, 0, 1, 2, 2)
        
        # Block 22: 1x1x2 (Schoolbook)
        U22, V22, W22 = schoolbook_112()
        U_332, V_332, W_332 = add_block(U22, V22, W22, 2, 2, 0, 1, 1, 2)
        
        return U_332, V_332, W_332

    # Get decompositions
    U_333, V_333, W_333 = get_333_decomp()
    U_332, V_332, W_332 = get_332_decomp()
    
    # Combine for M_{3,3,8}
    # Block 1: k=0,1,2 (M_{3,3,3})
    # Block 2: k=3,4,5 (M_{3,3,3})
    # Block 3: k=6,7 (M_{3,3,2})
    
    U_full = np.zeros((9, 0))
    V_full = np.zeros((24, 0))
    W_full = np.zeros((24, 0))
    
    # Block 1
    U_full, V_full, W_full = add_block(U_333, V_333, W_333, 0, 0, 0, 3, 3, 3)
    # Block 2
    U_full, V_full, W_full = add_block(U_333, V_333, W_333, 0, 0, 3, 3, 3, 3)
    # Block 3
    U_full, V_full, W_full = add_block(U_332, V_332, W_332, 0, 0, 6, 3, 3, 2)
    
    # Verify
    error = 0
    for u in range(9):
        i_u = u // 3
        j_u = u % 3
        for v in range(24):
            j_v = v // 8
            k_v = v % 8
            if j_u != j_v:
                continue
            for w in range(24):
                k_w = w // 3
                i_w = w % 3
                if i_u == i_w and k_v == k_w:
                    target = 1.0
                else:
                    target = 0.0
                s_val = np.sum(U_full[u, :] * V_full[v, :] * W_full[w, :])
                error += (s_val - target) ** 2
                
    if error < 1e-9:
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    else:
        # Fallback
        U_fb = np.zeros((9, 72))
        V_fb = np.zeros((24, 72))
        W_fb = np.zeros((24, 72))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(8):
                    U_fb[i*3+j, col] = 1
                    V_fb[j*8+k, col] = 1
                    W_fb[k*3+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
