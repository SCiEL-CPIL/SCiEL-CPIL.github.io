import time

def construct():
    n, m, p = 2, 3, 4
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 2x1x4 (Rank 8)
    # A: 2x1 (indices 0,1 in A)
    # B: 1x4 (indices 0,1,2,3 in B)
    # C: 2x4 (indices k*2+i in C)
    for i in range(2):
        for k in range(4):
            t = i * 4 + k
            U[i * 3 + 0][t] = 1
            V[0 * 4 + k][t] = 1
            W[k * 2 + i][t] = 1
            
    # Part 2: 2x2x4 (Rank 14)
    # A: 2x2 (indices 3,4,6,7 in A? No. i*3+j. j=1,2. i=0,1. 0*3+1=1, 0*3+2=2, 1*3+1=4, 1*3+2=5)
    # Wait. A is 2x3.
    # Part 1 used j=0. Indices 0, 1.
    # Part 2 uses j=1,2. Indices 1,2 (for i=0) and 4,5 (for i=1).
    # B: 2x4 (j=1,2). Indices j*4+k.
    # j=1: 4,5,6,7. j=2: 8,9,10,11.
    # C: 2x4. Indices k*2+i.
    
    # Strassen 2x2x2 coefficients
    # U_s: 2x7. V_s: 2x7. W_s: 2x7 (flattened 2x2)
    # Indices for Strassen A: 0,1. B: 0,1. C: 0,1.
    
    U_s = [
        [1, 0, 1, 0, 1, -1, 1], # a
        [0, 0, 0, 0, 1, 0, -1]  # b
    ]
    # Wait, d is A[1,1]. c is A[1,0].
    # My U_s above only has 2 rows. I need 4 rows for 2x2?
    # No, U is (n*m) x rank. For 2x2, n*m=4.
    # Rows: a(0,0), b(0,1), c(1,0), d(1,1).
    
    U_s = [
        [1, 0, 1, 0, 1, -1, 1], # a (0,0)
        [0, 0, 0, 0, 1, 0, -1], # b (0,1)
        [0, 1, 0, 0, 0, 1, 0],  # c (1,0)
        [1, 1, 0, 1, 0, 0, -1]  # d (1,1)
    ]
    
    V_s = [
        [1, 1, 0, -1, 0, 1, 1], # e (0,0)
        [0, 0, 1, 0, 0, 1, 0],  # f (0,1)
        [0, 0, 0, 1, 0, 0, 1],  # g (1,0)
        [1, 0, -1, 0, 1, 0, 1]  # h (1,1)
    ]
    
    # W_s: 4x7. C indices: p(0,0), q(0,1), r(1,0), s(1,1).
    # p = m1 + m4 - m5 + m7
    # q = m3 + m5
    # r = m2 + m4
    # s = m1 - m2 + m3 + m6
    W_s = [
        [1, 0, 0, 1, -1, 0, 1], # p
        [0, 0, 1, 0, 1, 0, 0],  # q
        [0, 1, 0, 1, 0, 0, 0],  # r
        [1, -1, 1, 0, 0, 1, 0]  # s
    ]
    
    # Scalar 1x1x2 coefficients
    # U_sc: 1x2. [1, 1]
    # V_sc: 1x2. [[1, 0], [0, 1]]? No, V is (m*p) x rank.
    # For 1x1x2, m=1, p=2. V is 1x2 x 2.
    # Term 0: V[0,0]=1. Term 1: V[0,1]=1.
    V_sc = [
        [1, 0],
        [0, 1]
    ]
    # W_sc: 2x2.
    # Term 0: W[0,0]=1. Term 1: W[1,0]=1?
    # C is 1x2. Indices 0,1.
    W_sc = [
        [1, 0],
        [0, 1]
    ]
    
    for s_idx in range(7):
        for q_idx in range(2):
            t = 8 + s_idx * 2 + q_idx
            
            # A mapping: i in 0..1, j_local in 0..1 -> j_global = 1 + j_local
            # U_s row index: i*2 + j_local
            for i in range(2):
                for j_local in range(2):
                    val_u = U_s[i * 2 + j_local][s_idx]
                    val_sc = 1 # U_sc is always 1
                    if val_u != 0:
                        row_a = i * 3 + (1 + j_local)
                        U[row_a][t] += val_u * val_sc
                        
            # B mapping: j_local in 0..1 -> j_global = 1 + j_local
            # k_local in 0..1, k_q in 0..1 -> k_global = 2*k_local + k_q
            # V_s row index: j_local*2 + k_local
            # V_sc row index: k_q (since m_sc=1, index is k_q)
            for j_local in range(2):
                for k_local in range(2):
                    val_v = V_s[j_local * 2 + k_local][s_idx]
                    if val_v != 0:
                        # V_sc depends on q_idx
                        # V_sc[k_q, q_idx]
                        # q_idx is 0 or 1. k_q must match q_idx for non-zero?
                        # V_sc[0,0]=1, V_sc[0,1]=0. V_sc[1,0]=0, V_sc[1,1]=1.
                        # So val_sc is 1 if k_q == q_idx, else 0.
                        # We iterate k_q? No, V_sc is fixed for the term.
                        # The term q_idx corresponds to k_q = q_idx.
                        # So we only add to the B index corresponding to k_q = q_idx.
                        k_q = q_idx
                        k_global = 2 * k_local + k_q
                        row_b = (1 + j_local) * 4 + k_global
                        V[row_b][t] += val_v * 1 # val_sc is 1 for the active k_q
                        
            # C mapping: i in 0..1, k_local in 0..1, k_q in 0..1
            # W_s row index: i*2 + k_local
            # W_sc row index: k_q
            for i in range(2):
                for k_local in range(2):
                    val_w = W_s[i * 2 + k_local][s_idx]
                    if val_w != 0:
                        k_q = q_idx
                        k_global = 2 * k_local + k_q
                        row_c = k_global * 2 + i
                        W[row_c][t] += val_w * 1
                        
    return {"U": U, "V": V, "W": W}
