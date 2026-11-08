import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 5.0
    
    n, m, p = DIMENSIONS
    
    # Rank 60 decomposition
    # M(3,3,7) = M(3,2,7) + M(3,1,7)
    # M(3,2,7) = M(2,2,7) + M(1,2,7)
    # M(2,2,7) = 3*M(2,2,2) + M(2,2,1) -> 3*7 + 4 = 25
    # M(1,2,7) -> 14
    # M(3,1,7) -> 21
    # Total 60
    
    rank = 60
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_rank1_term(U_vec, V_vec, W_vec):
        nonlocal col
        if col >= rank:
            return
        U[:, col] = U_vec
        V[:, col] = V_vec
        W[:, col] = W_vec
        col += 1

    def u_idx(i, j): return i * m + j
    def v_idx(i, j): return i * p + j
    def w_idx(i, j): return j * n + i

    def add_strassen(a_base_i, b_base_i, c_base_i, k_base):
        a00 = u_idx(a_base_i, 0)
        a01 = u_idx(a_base_i, 1)
        a10 = u_idx(a_base_i + 1, 0)
        a11 = u_idx(a_base_i + 1, 1)
        
        b00 = v_idx(b_base_i, k_base)
        b01 = v_idx(b_base_i, k_base + 1)
        b10 = v_idx(b_base_i + 1, k_base)
        b11 = v_idx(b_base_i + 1, k_base + 1)
        
        c00 = w_idx(c_base_i, k_base)
        c01 = w_idx(c_base_i, k_base + 1)
        c10 = w_idx(c_base_i + 1, k_base)
        c11 = w_idx(c_base_i + 1, k_base + 1)
        
        # m1: (a00+a11)(b00+b11)
        U1 = np.zeros(n*m); V1 = np.zeros(m*p); W1 = np.zeros(p*n)
        U1[a00] = 1; U1[a11] = 1
        V1[b00] = 1; V1[b11] = 1
        W1[c00] = 1; W1[c11] = 1
        add_rank1_term(U1, V1, W1)
        
        # m2: (a10+a11)b00
        U2 = np.zeros(n*m); V2 = np.zeros(m*p); W2 = np.zeros(p*n)
        U2[a10] = 1; U2[a11] = 1
        V2[b00] = 1
        W2[c10] = 1; W2[c11] = -1
        add_rank1_term(U2, V2, W2)
        
        # m3: a00(b01-b11)
        U3 = np.zeros(n*m); V3 = np.zeros(m*p); W3 = np.zeros(p*n)
        U3[a00] = 1
        V3[b01] = 1; V3[b11] -1 if False else 0
        V3[b11] = -1
        W3[c01] = 1; W3[c11] = 1
        add_rank1_term(U3, V3, W3)
        
        # m4: a11(b10-b00)
        U4 = np.zeros(n*m); V4 = np.zeros(m*p); W4 = np.zeros(p*n)
        U4[a11] = 1
        V4[b10] = 1; V4[b00] = -1
        W4[c00] = 1; W4[c10] = 1
        add_rank1_term(U4, V4, W4)
        
        # m5: (a00+a01)b11
        U5 = np.zeros(n*m); V5 = np.zeros(m*p); W5 = np.zeros(p*n)
        U5[a00] = 1; U5[a01] = 1
        V5[b11] = 1
        W5[c00] = -1; W5[c01] = 1
        add_rank1_term(U5, V5, W5)
        
        # m6: (a10-a00)(b00+b01)
        U6 = np.zeros(n*m); V6 = np.zeros(m*p); W6 = np.zeros(p*n)
        U6[a10] = 1; U6[a00] = -1
        V6[b00] = 1; V6[b01] = 1
        W6[c11] = 1
        add_rank1_term(U6, V6, W6)
        
        # m7: (a01-a11)(b10+b11)
        U7 = np.zeros(n*m); V7 = np.zeros(m*p); W7 = np.zeros(p*n)
        U7[a01] = 1; U7[a11] = -1
        V7[b10] = 1; V7[b11] = 1
        W7[c00] = 1
        add_rank1_term(U7, V7, W7)

    def add_schoolbook_2x2x1(a_base_i, b_base_i, c_base_i, k_base):
        for i in range(2):
            for j in range(2):
                U_term = np.zeros(n*m); V_term = np.zeros(m*p); W_term = np.zeros(p*n)
                U_term[u_idx(a_base_i + i, j)] = 1
                V_term[v_idx(b_base_i + j, k_base)] = 1
                W_term[w_idx(c_base_i + i, k_base)] = 1
                add_rank1_term(U_term, V_term, W_term)

    # 1. M(2,2,7) part: A[0:2, 0:2] * B[0:2, 0:7] -> C[0:2, 0:7]
    add_strassen(0, 0, 0, 0)   # C[0:2, 0:2]
    add_strassen(0, 0, 0, 2)   # C[0:2, 2:4]
    add_strassen(0, 0, 0, 4)   # C[0:2, 4:6]
    add_schoolbook_2x2x1(0, 0, 0, 6)  # C[0:2, 6:7]
    
    # 2. M(1,2,7) part: A[2, 0:2] * B[0:2, 0:7] -> C[2, 0:7]
    for j in range(2):
        for k in range(7):
            U_term = np.zeros(n*m); V_term = np.zeros(m*p); W_term = np.zeros(p*n)
            U_term[u_idx(2, j)] = 1
            V_term[v_idx(j, k)] = 1
            W_term[w_idx(2, k)] = 1
            add_rank1_term(U_term, V_term, W_term)

    # 3. M(3,1,7) part: A[0:3, 2] * B[2, 0:7] -> C[0:3, 0:7]
    for i in range(3):
        for k in range(7):
            U_term = np.zeros(n*m); V_term = np.zeros(m*p); W_term = np.zeros(p*n)
            U_term[u_idx(i, 2)] = 1
            V_term[v_idx(2, k)] = 1
            W_term[w_idx(i, k)] = 1
            add_rank1_term(U_term, V_term, W_term)

    # Verification
    if time.time() + 10.0 < deadline - safety_margin:
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_true = A_test @ B_test
        C_comp = np.zeros((n, p))
        
        U_a = U
        V_a = V
        W_a = W
        
        for i in range(n):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    u_vec = U_a[i*m:(i+1)*m, r]
                    v_vec = V_a[k::p, r]
                    val += W_a[k*n+i, r] * np.dot(u_vec, A_test[i, :]) * np.dot(v_vec, B_test[:, k])
                C_comp[i, k] = val
                
        if not np.allclose(C_comp, C_true, atol=1e-9):
            # Fallback to schoolbook
            rank_fb = n * m * p
            U_fb = np.zeros((n * m, rank_fb))
            V_fb = np.zeros((m * p, rank_fb))
            W_fb = np.zeros((p * n, rank_fb))
            col_fb = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j, col_fb] = 1
                        V_fb[j * p + k, col_fb] = 1
                        W_fb[k * n + i, col_fb] = 1
                        col_fb += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
