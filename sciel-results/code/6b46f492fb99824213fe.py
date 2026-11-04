import numpy as np

DIMENSIONS = [4, 4, 9]

def _strassen_2x2_vectors():
    """
    Returns lists U, V, W of 7 vectors each of length 4.
    u index: i*2 + j
    v index: j*2 + k
    w index: k*2 + i
    """
    U = []
    V = []
    W = []
    
    # M1 = (A11+A22)(B11+B22)
    # A11+A22: u[0]=1, u[3]=1
    # B11+B22: v[0]=1, v[3]=1
    # C11+=1, C22+=1: w[0]=1, w[3]=1
    u = [0.0]*4; u[0]=1; u[3]=1
    v = [0.0]*4; v[0]=1; v[3]=1
    w = [0.0]*4; w[0]=1; w[3]=1
    U.append(u); V.append(v); W.append(w)
    
    # M2 = (A21+A22)B11
    # A21+A22: u[2]=1, u[3]=1
    # B11: v[0]=1
    # C21+=1: w[2]=1 (k=1, i=0 -> 1*2+0=2)
    u = [0.0]*4; u[2]=1; u[3]=1
    v = [0.0]*4; v[0]=1
    w = [0.0]*4; w[2]=1
    U.append(u); V.append(v); W.append(w)
    
    # M3 = A11(B12-B22)
    # A11: u[0]=1
    # B12-B22: v[1]=1, v[3]=-1
    # C12+=1: w[1]=1 (k=0, i=1 -> 0*2+1=1)
    u = [0.0]*4; u[0]=1
    v = [0.0]*4; v[1]=1; v[3]=-1
    w = [0.0]*4; w[1]=1
    U.append(u); V.append(v); W.append(w)
    
    # M4 = A22(B21-B11)
    # A22: u[3]=1
    # B21-B11: v[2]=1, v[0]=-1
    # C11+=1, C21+=1: w[0]=1, w[2]=1
    u = [0.0]*4; u[3]=1
    v = [0.0]*4; v[2]=1; v[0]=-1
    w = [0.0]*4; w[0]=1; w[2]=1
    U.append(u); V.append(v); W.append(w)
    
    # M5 = (A11+A12)B22
    # A11+A12: u[0]=1, u[1]=1
    # B22: v[3]=1
    # C11-=1, C12+=1: w[0]=-1, w[1]=1
    u = [0.0]*4; u[0]=1; u[1]=1
    v = [0.0]*4; v[3]=1
    w = [0.0]*4; w[0]=-1; w[1]=1
    U.append(u); V.append(v); W.append(w)
    
    # M6 = (A21-A11)(B11+B12)
    # A21-A11: u[2]=1, u[0]=-1
    # B11+B12: v[0]=1, v[1]=1
    # C21+=1: w[2]=1
    u = [0.0]*4; u[2]=1; u[0]=-1
    v = [0.0]*4; v[0]=1; v[1]=1
    w = [0.0]*4; w[2]=1
    U.append(u); V.append(v); W.append(w)
    
    # M7 = (A12-A22)(B21+B22)
    # A12-A22: u[1]=1, u[3]=-1
    # B21+B22: v[2]=1, v[3]=1
    # C22+=1: w[3]=1
    u = [0.0]*4; u[1]=1; u[3]=-1
    v = [0.0]*4; v[2]=1; v[3]=1
    w = [0.0]*4; w[3]=1
    U.append(u); V.append(v); W.append(w)
    
    return U, V, W

def _strassen_4x4_decomposition():
    """
    Returns U, V, W lists for 4x4x4 matrix multiplication using Strassen.
    Rank should be 49.
    """
    S_U, S_V, S_W = _strassen_2x2_vectors()
    
    # Strassen coefficients for 4x4
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M7
    
    # alpha[k][i_b][j_b]
    # beta[k][j_b][k_b]
    # gamma[k][k_b][i_b] -> Note: C block is C[i_b, k_b]. 
    # In my previous derivation, I used gamma[i_b, k_b].
    # Let's stick to: C[i_b, k_b] gets contribution from M_k with coeff gamma[k][i_b][k_b].
    
    # Note: In 4x4, blocks are 2x2.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Indices: i_b, j_b, k_b in {0,1}
    
    alphas = [
        # M1
        [[1, 0], [0, 1]], 
        # M2
        [[0, 0], [1, 1]], 
        # M3
        [[1, 0], [0, 0]], 
        # M4
        [[0, 0], [0, 1]], 
        # M5
        [[1, 1], [0, 0]], 
        # M6
        [[-1, 0], [1, 0]], 
        # M7
        [[0, 1], [0, -1]]
    ]
    
    betas = [
        # M1
        [[1, 0], [0, 1]], 
        # M2
        [[1, 0], [0, 0]], 
        # M3
        [[0, 1], [0, -1]], 
        # M4
        [[-1, 0], [1, 0]], 
        # M5
        [[0, 0], [0, 1]], 
        # M6
        [[1, 1], [0, 0]], 
        # M7
        [[0, 0], [1, 1]]
    ]
    
    gammas = [
        # M1: C00+=1, C11+=1
        [[1, 0], [0, 1]], 
        # M2: C10+=1
        [[0, 0], [1, 0]], 
        # M3: C01+=1
        [[0, 1], [0, 0]], 
        # M4: C00+=1
        [[1, 0], [0, 0]], 
        # M5: C00-=1, C01+=1
        [[-1, 1], [0, 0]], 
        # M6: C10+=1
        [[0, 0], [1, 0]], 
        # M7: C00-=1, C11+=1
        [[-1, 0], [0, 1]]
    ]
    
    U_list = []
    V_list = []
    W_list = []
    
    for k in range(7):
        for i_b in range(2):
            for j_b in range(2):
                a_coeff = alphas[k][i_b][j_b]
                if a_coeff == 0: continue
                for k_b in range(2):
                    b_coeff = betas[k][j_b][k_b]
                    if b_coeff == 0: continue
                    c_coeff = gammas[k][i_b][k_b]
                    if c_coeff == 0: continue
                    
                    # Combine coefficients
                    coeff = a_coeff * b_coeff * c_coeff
                    if coeff == 0: continue
                    
                    for t in range(7):
                        u_vec = np.zeros(16)
                        v_vec = np.zeros(16)
                        w_vec = np.zeros(16)
                        
                        # Fill U
                        # Block (i_b, j_b)
                        # Indices in 4x4: i = 2*i_b + i_l, j = 2*j_b + j_l
                        # U index: i*4 + j
                        for i_l in range(2):
                            for j_l in range(2):
                                idx_u = (2*i_b + i_l)*4 + (2*j_b + j_l)
                                idx_su = i_l*2 + j_l
                                u_vec[idx_u] = coeff * S_U[t][idx_su]
                                
                        # Fill V
                        # Block (j_b, k_b)
                        # Indices in 4x4: j = 2*j_b + j_l, k = 2*k_b + k_l
                        # V index: j*4 + k
                        for j_l in range(2):
                            for k_l in range(2):
                                idx_v = (2*j_b + j_l)*4 + (2*k_b + k_l)
                                idx_sv = j_l*2 + k_l
                                v_vec[idx_v] = coeff * S_V[t][idx_sv]
                                
                        # Fill W
                        # Block (k_b, i_b) -> C[i_b, k_b]
                        # Indices in 4x4: k = 2*k_b + k_l, i = 2*i_b + i_l
                        # W index: k*4 + i
                        for k_l in range(2):
                            for i_l in range(2):
                                idx_w = (2*k_b + k_l)*4 + (2*i_b + i_l)
                                idx_sw = k_l*2 + i_l
                                w_vec[idx_w] = coeff * S_W[t][idx_sw]
                                
                        U_list.append(u_vec)
                        V_list.append(v_vec)
                        W_list.append(w_vec)
                        
    return U_list, V_list, W_list

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    U_all = []
    V_all = []
    W_all = []
    
    # Part 1: 4x4 x 4x4 (first 4 columns of B)
    # Rank 49
    S4_U, S4_V, S4_W = _strassen_4x4_decomposition()
    
    # Part 2: 4x4 x 4x4 (next 4 columns of B, cols 4-7)
    # Rank 49
    # We can reuse the same decomposition, just shift the V and W indices.
    # V index for col offset 4: j*p + (4+k)
    # W index for col offset 4: (4+k)*n + i
    
    # Part 3: 4x4 x 4x1 (last column of B, col 8)
    # Rank 16 (Schoolbook)
    
    # Helper to add term with offset
    def add_term(u, v, w, v_offset=0, w_offset=0):
        # v_offset is added to the column index of B (which is the second part of V index)
        # V index: j*p + k. If we shift k by v_offset, v[j*p + (k+v_offset)]
        # But our v_vec is built for k in 0..3.
        # So we need to create a new v_vec of length 36.
        # Similarly for w_vec of length 36.
        
        v_full = np.zeros(m*p)
        w_full = np.zeros(p*n)
        
        for idx, val in enumerate(v):
            if val != 0:
                j = idx // 4
                k = idx % 4
                new_k = k + v_offset
                if 0 <= new_k < p:
                    v_full[j*p + new_k] = val
                    
        for idx, val in enumerate(w):
            if val != 0:
                k = idx // 4
                i = idx % 4
                new_k = k + w_offset
                if 0 <= new_k < p:
                    w_full[new_k*n + i] = val
                    
        U_all.append(u)
        V_all.append(v_full)
        W_all.append(w_full)

    # Add Strassen terms for first 4 cols
    for i in range(len(S4_U)):
        add_term(S4_U[i], S4_V[i], S4_W[i], v_offset=0, w_offset=0)
        
    # Add Strassen terms for next 4 cols
    for i in range(len(S4_U)):
        add_term(S4_U[i], S4_V[i], S4_W[i], v_offset=4, w_offset=4)
        
    # Add Schoolbook for last 1 col
    for i in range(n):
        for j in range(m):
            for k in range(1): # k=0
                u = np.zeros(n*m)
                v = np.zeros(m*p)
                w = np.zeros(p*n)
                u[i*m + j] = 1.0
                v[j*p + (8 + k)] = 1.0
                w[(8 + k)*n + i] = 1.0
                U_all.append(u)
                V_all.append(v)
                W_all.append(w)
                
    rank = len(U_all)
    
    # Convert to list of lists
    U = np.array(U_all).T
    V = np.array(V_all).T
    W = np.array(W_all).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
